import os
import argparse
from pathlib import Path
import pandas as pd
from scipy.io import arff
import torch
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from utils import (
    update_dataset_registry,
    estimate_seasonality_generic,
    plot_mean_series_with_period,
    encode_in_batches,
    plot_latent_space,
)
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score, f1_score
from sklearn.preprocessing import LabelEncoder

PATH = Path(__file__).parent.absolute()

def parse_args():
    parser = argparse.ArgumentParser(description='Run clustering pipeline for UCR/UEA datasets.')
    parser.add_argument(
        '--dataset-position',
        type=int,
        default=None,
        help='Dataset index in the sorted file list (used when --dataset-name is not provided).',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help='Dataset folder name. Overrides --dataset-position if provided.',
    )
    parser.add_argument(
        '--launch',
        type=str,
        default='FMMVCC',
        choices=['FMMVCC'],
        help='Training launch method.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='unidirectional',
        choices=['unidirectional', 'bidirectional'],
        help='Encoder mode for FMMVCC.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dims', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretraining-epoch', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--m', type=float, default=1.5)
    parser.add_argument('--skip-seasonality', action='store_true', help='Skip seasonality analysis and plotting.')
    parser.add_argument(
        '--plot-root',
        type=Path,
        default=Path(PATH / 'plot'),
        help='Root directory where plots and summary metrics are saved.',
    )
    return parser.parse_args()


def select_dataset(file_list, dataset_name=None, dataset_position=None):
    if not file_list:
        raise ValueError('No datasets found in the configured data path.')

    if dataset_name:
        # Match dataset names in a case-insensitive way for convenience.
        dataset_map = {name.lower(): name for name in file_list}
        selected = dataset_map.get(dataset_name.lower())
        if selected is None:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available dataset count: {len(file_list)}"
            )
        return selected, file_list.index(selected)

    # If no explicit position is provided, auto-select by available dataset names.
    if dataset_position is None:
        if len(file_list) == 1:
            return file_list[0], 0
        print(
            'No dataset position provided. Multiple datasets found; '
            f"using the first one alphabetically: '{file_list[0]}'."
        )
        dataset_position = 0

    if dataset_position < 0 or dataset_position >= len(file_list):
        print(
            f"dataset_position={dataset_position} is out of range [0, {len(file_list) - 1}]. "
            'Falling back to the first dataset.'
        )
        dataset_position = 0

    return file_list[dataset_position], dataset_position


def main():
    args = parse_args()

    data_path = Path(__file__).parent.absolute() / 'tsc_datasets' / 'extracted'
    data_path = data_path / 'Univariate2018_arff' / 'Univariate_arff'
    # Dataset names are expected to be folder names under the UCR root.
    file_list = sorted([
        p.name for p in data_path.iterdir() if p.is_dir()
    ])
    print(f"Found {len(file_list)} datasets")
    selected_file, file_position = select_dataset(
        file_list,
        dataset_name=args.dataset_name,
        dataset_position=args.dataset_position,
    )
    print("Selected file:", selected_file)

    torch.cuda.empty_cache()
    full_path = data_path / selected_file
    print("Complete path of the selected file:", full_path)

    # Load data from ARFF files
    file_path_train = full_path / f"{selected_file}_TRAIN.arff"
    file_path_test = full_path / f"{selected_file}_TEST.arff"
    train, _ = arff.loadarff(file_path_train)
    test, _ = arff.loadarff(file_path_test)
    train_dataset = pd.DataFrame(train).fillna(0)
    test_dataset = pd.DataFrame(test).fillna(0)

    print(train_dataset.head())

    # Separate features and labels, encode labels, and count clusters
    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values
    X_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values
    y_train = np.array(y_train).astype(str)
    y_test = np.array(y_test).astype(str)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Count how many unique clusters are present
    unique_clusters = set(y_train)
    n_clusters = len(unique_clusters)

    # Update dataset registry
    json_path = Path(__file__).parent.absolute() / "dataset_registry.json"
    update_dataset_registry(
        json_path=json_path,
        dataset_name=selected_file,
        position=file_position,
        univariate=True,
        n_clusters=n_clusters,
        train_shape=X_train.shape[0],
        temporal_length=X_train.shape[1],
    )

    # Normalize time series
    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f'The shape of X_train_scaled is: {X_train_scaled.shape}')
    print(f'The shape of X_test_scaled is: {X_test_scaled.shape}')

    # Setting metrics variables
    ari = None
    ri = None
    nmi = None
    f1 = None

    # Seasonality analysis
    if not args.skip_seasonality:
        seasonality_period = estimate_seasonality_generic(X_train_scaled)
        if seasonality_period is not None:
            print(f"Estimated seasonality period: {seasonality_period}")
            plot_mean_series_with_period(X_train_scaled[:, :seasonality_period * 20], seasonality_period)
        else:
            raise ValueError(
                'No significant seasonality detected. Use --skip-seasonality to bypass this step.'
            )

    if args.launch == 'FMMVCC':
        print("Using FMMVCC")
        from batch_run import run_FMMVCC

        run_label = 'FMMVCC' if args.mode == 'unidirectional' else f'FMMVCC_{args.mode}'
        config = {
            'batch_size': args.batch_size,
            'output_dims': args.output_dims,
            'lr': args.lr,
            'pretraining_epoch': args.pretraining_epoch,
            'MaxIter': args.max_iter,
            'm': args.m,
        }
        acc, nmi, ari, ri, fmi, f1, model = run_FMMVCC(
            X_train,
            X_test,
            y_train,
            y_test,
            selected_file,
            config,
            args.mode,
        )

        print(f"Results: acc={acc}, nmi={nmi}, ari={ari}, ri={ri}, fmi={fmi}, f1={f1}")

        # Folder in which predictions are saved
        results_dir_name = 'results' if args.mode == 'unidirectional' else f'results_{args.mode}'
        results_folder = Path.cwd() / results_dir_name / selected_file / 'label'

        # Verify files exist before reading
        y_true_path = results_folder / f"{selected_file}_label_true.csv"
        y_pred_path = results_folder / f"{selected_file}_label_pred.csv"
        if not y_true_path.exists() or not y_pred_path.exists():
            raise FileNotFoundError(
                f"Prediction files not found in {results_folder}."
            )

        y_pred_train = pd.read_csv(y_true_path)['label_true'].values
        y_pred_test = pd.read_csv(y_pred_path)['label_pred'].values
        y_pred_train = np.array(y_pred_train).astype(int)
        y_pred_test = np.array(y_pred_test).astype(int)
        y_pred_train = y_pred_train[:len(y_train)]
        y_pred_test = y_pred_test[:len(y_test)]

        # Plot of latent space
        latent_plot_subdir = run_label + '/' + selected_file
        u1 = encode_in_batches(model, X_train_scaled)
        plot_latent_space(u1, y_train, latent_plot_subdir, 'Training Data', args.plot_root)

        u2 = encode_in_batches(model, X_test_scaled)
        plot_latent_space(u2, y_test, latent_plot_subdir, 'Test Data', args.plot_root)
    else:
        raise ValueError(f"Unknown launch method: {args.launch}")

    # Metrics
    ari = adjusted_rand_score(y_test, y_pred_test) if ari is None else ari
    ri = rand_score(y_test, y_pred_test) if ri is None else ri
    nmi = normalized_mutual_info_score(y_test, y_pred_test) if nmi is None else nmi
    f1 = f1_score(y_test, y_pred_test, average='macro') if f1 is None else f1

    launch_name = 'FMMVCC' if args.mode == 'unidirectional' else f'FMMVCC_{args.mode}'
    results = {
        "ARI": ari,
        "RI": ri,
        "NMI": nmi,
        "F1": f1,
        "Data set": f'Position {file_position} - Name {selected_file}',
    }
    # Save results to a file in the plot folder
    results_path = args.plot_root / launch_name / selected_file
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / 'clustering_results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print("Finished clustering and plotting.")


if __name__ == '__main__':
    main()
