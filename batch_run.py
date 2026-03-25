import os
import numpy as np
import torch
from fmmvcc import FMMVCC_Model
import time
import datetime
import logging
import datautils

def setup_logger(logger_name, log_file, flag=None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s : %(message)s')

    # No duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- LOG FILE HANDLING ---
    if flag is None or flag == 'pretrain':
        file_mode = 'w'
    elif flag == 'finetune_only':
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                if "Starting fine-tuning" in line:
                    break
                cleaned_lines.append(line)
            with open(log_file, 'w') as f:
                f.writelines(cleaned_lines)
        file_mode = 'a'
    elif flag == 'finished':
        file_mode = 'a'
    else:
        file_mode = 'a'

    # File handler
    fileHandler = logging.FileHandler(log_file, mode=file_mode)
    fileHandler.setFormatter(formatter)

    # Stream handler
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


def train_and_evaluate_from_loader(train_loader, test_loader, config, logger):
    """
    Function to train and evaluate FCACC from already created DataLoaders.
    Inputs:
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data
    - config: model configuration (batch size, number of epochs, etc.)
    - logger: logger used to log information and execution details
    Returns:
    - acc, nmi, ari, ri, fmi, f1: clustering metrics on the test set
    - model: trained Mamba_FCC model
    """
    dataset_dir = f"launches_{config['mode']}/{config['dataset_name']}" if config.get('mode', 'unidirectional') != 'unidirectional' else f"launches/{config['dataset_name']}"
    pretrain_path = os.path.join(dataset_dir, "Pretraining_phase.pt")
    finetune_path = os.path.join(dataset_dir, "Finetuning_phase.pt")
    centers_path = os.path.join(dataset_dir, "Centers.pt")

    logger.info(f"Running FCACC on dataset: {config['dataset_name']}")

    # Creation of the model
    model = FMMVCC_Model(train_loader, **config)

    t0 = time.time()

    # Pretraining & Fine-tuning
    if config.get('MaxIter', 0) != 0:
        if os.path.exists(finetune_path):
            # Upload trained model and centers, skip pretraining and fine-tuning
            state_dict = torch.load(
                finetune_path,
                map_location=model.device
            )
            model.load_state_dict(state_dict, strict=False)
            
            model.u_mean = torch.load(
                centers_path,
                map_location=model.device
            )
            
            from fmmvcc import MultiViewEncoder
            model.encoder_module = MultiViewEncoder(
                model.view_encoders,
                model.view_decoders,
                model.cross_view_decoders
            )
            model.__dict__['net'] = torch.optim.swa_utils.AveragedModel(model.encoder_module)
            model.__dict__['net'].update_parameters(model.encoder_module)
            
        elif os.path.exists(pretrain_path):
            # Upload pretrained model, skip pretraining and do fine-tuning
            logger.info("Starting fine-tuning...")
            state_dict = torch.load(
                pretrain_path,
                map_location=model.device
            )
            model.load_state_dict(state_dict, strict=False)
            
            from fmmvcc import MultiViewEncoder
            model.encoder_module = MultiViewEncoder(
                model.view_encoders,
                model.view_decoders,
                model.cross_view_decoders
            )
            model.__dict__['net'] = torch.optim.swa_utils.AveragedModel(model.encoder_module)
            model.__dict__['net'].update_parameters(model.encoder_module)
            
            model.Finetuning(logger)
        else:
            # Training from scratch
            logger.info("Starting pretraining...")
            model.Pretraining(logger)

            logger.info("Starting fine-tuning...")
            model.Finetuning(logger)

    t = time.time() - t0
    logger.info(f"Training time: {datetime.timedelta(seconds=t)}\n")
    
    # Evaluation on test set
    acc, nmi, ari, ri, fmi, f1 = model.eval_with_test_data(
        config['dataset_name'], 
        logger, 
        test_loader)

    logger.info(f"Test results: acc={acc}, nmi={nmi}, ari={ari}, ri={ri}, fmi={fmi}, f1={f1}")
    return acc, nmi, ari, ri, fmi, f1, model

def run_FMMVCC(X_train, X_test, label_train, label_test, name, config, mode='unidirectional'):
    """
    Inputs:
    - X_train: training data
    - X_test: test data
    - label_train: labels of the training data
    - label_test: labels of the test data
    - config: model configuration (batch size, number of epochs, etc.)
    - mode: "unidirectional", "bidirectional" or "causalconv"

    Returns:
    - acc, nmi, ari, ri, fmi, f1: clustering metrics on the test set
    - model: trained FMMVCC model
    """

    # Create the data loader (univariate)
    if X_train.ndim == 2:
        X_train = X_train[..., np.newaxis]
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    
    train_index = np.arange(X_train.shape[0])
    test_index = np.arange(X_test.shape[0])

    train_loader, test_loader = datautils.create_data_loader(X_train, X_test, label_train, label_test, train_index, test_index,  config['batch_size'])

    # Configurazione
    config['dataset_name'] = name
    config['dataset_size'] = X_train.shape[0] + X_test.shape[0]
    config['timesteps_len'] = X_train.shape[1]
    config['input_dims'] = X_train.shape[2]
    config['n_cluster'] = len(np.unique(label_train)) 
    config['mode'] = mode
    
    # Create launches directory if not exists
    launches_dir = os.path.join(os.getcwd(), f'launches_{mode}' if mode != 'unidirectional' else 'launches')
    if not os.path.exists(launches_dir):
        os.makedirs(launches_dir)
         
    dataset_launch_dir = os.path.join(launches_dir, name)
    if not os.path.exists(dataset_launch_dir):
        os.makedirs(dataset_launch_dir)

    # Setup logger
    log_file = os.path.join(dataset_launch_dir, f'{name}_log.txt')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            if "Starting pretraining" in log_content and "Starting fine-tuning" not in log_content:
                flag = 'pretrain'
            elif "Starting fine-tuning" in log_content and "Test results" not in log_content:
                flag = 'finetune_only'
            elif "Test results" in log_content:
                flag = 'finished'
            else:
                flag = None
    else:
        flag = None
    logger = setup_logger(f'{name}_logger', log_file, flag)
    logger.info(f"Configuration: {config}")
    try:
        return train_and_evaluate_from_loader(train_loader, test_loader, config, logger)
    except Exception as e:
        logger.error("An error occurred during training/evaluation", exc_info=True)
        raise e