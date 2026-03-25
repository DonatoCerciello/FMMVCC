import math
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import MambaEncoder
from models.Metrics import acc,rand_index_score
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fmi
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import f1_score as f1
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss

from tools.tool import generate_pos_neg_index, MASK


class MultiViewEncoder(nn.Module):
    """Wrapper contenente solo gli encoder e i decoder (senza logica di training)"""
    def __init__(self, view_encoders, view_decoders, cross_view_decoders):
        super().__init__()
        self.view_encoders = view_encoders
        self.view_decoders = view_decoders
        self.cross_view_decoders = cross_view_decoders


class FMMVCC_Model(nn.Module):
    ''' Fuzzy Mamba-based Multi-View Contrastive Clustering for Time Series '''

    def __init__(
            self,
            data_loader,
            dataset_size,
            timesteps_len,
            batch_size,
            pretraining_epoch,
            n_cluster,
            dataset_name,
            input_dims,
            MaxIter=100,
            m=1.5,
            T1=2,
            output_dims=32,
            hidden_dims=64,
            n_layers=4,
            device='cuda',
            lr=0.001,
            max_train_length=4000,
            temporal_unit=0,
            mode='unidirectional',
            num_views = 4,
            ):

        super().__init__()
        self.device = device
        self.lr = lr
        self.num_cluster = n_cluster
        self.batch_size = batch_size
        self.T1 = T1
        self.m = m
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.timesteps_len = timesteps_len
        self.input_dims = input_dims
        self.dataset_name = dataset_name
        self.latent_size = output_dims
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.n_layers = n_layers
        self.mode = mode
        self.num_views = num_views
        self.hard_w  = min(1.0, self.T1 / 20)
        self.mask_mode = 0
        self.dropout = 0.2

        self.u_mean = torch.zeros([n_cluster, self.latent_size], device=self.device)
        
        # Encoders
        self.view_encoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_encoders[f'view_{i}'] = MambaEncoder(input_dims=input_dims, 
                                                        output_dims=self.latent_size, 
                                                        hidden_dims=hidden_dims, 
                                                        n_layers=self.n_layers,
                                                        mask_mode=self.mask_mode,
                                                        dropout=self.dropout,
                                                        mode=self.mode,
                                                        ).to(self.device)

        # Cross view decoders
        self.cross_view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                self.cross_view_decoders[f'{i}_to_{j}'] = nn.Sequential(
                    nn.Linear(self.latent_size, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, self.latent_size)
                )
                self.cross_view_decoders[f'{j}_to_{i}'] = nn.Sequential(
                    nn.Linear(self.latent_size, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, self.latent_size)
                )
        self.cross_view_decoders.to(self.device)

        # Intra view decoders
        self.view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_decoders[f'view_{i}_decoder'] = nn.Sequential(
                nn.Linear(self.latent_size, hidden_dims * 2),
                nn.LayerNorm(hidden_dims * 2),
                nn.ReLU(),
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_dims)
            )
        self.view_decoders.to(self.device)
        
        # Weights for losses
        self.log_w_contrast = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_cross = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_rec = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        self.log_w_msc = nn.Parameter(
            torch.zeros(1, device=self.device)
        )
        
        # Wrap in MultiViewEncoder and AveragedModel for SWA
        self.encoder_module = MultiViewEncoder(
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        )

        self.state_centers = nn.Parameter(
            F.normalize(
                torch.randn(n_cluster, self.latent_size, device=self.device),
                dim=1
            )
        )
        self.__dict__['net'] = torch.optim.swa_utils.AveragedModel(self.encoder_module)

    def Pretraining(self, logger):
        logger.info('Pretraining...')

        # Set all parameters to require gradients
        modules = [
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        ]

        for module in modules:
            module.train()
            for param in module.parameters():
                param.requires_grad = True

        optimizer = optim.AdamW(
            list(self.view_encoders.parameters()) +
            list(self.view_decoders.parameters()) +
            list(self.cross_view_decoders.parameters()),
            lr=self.lr
        )
        optimizer.add_param_group({'params': [self.log_w_contrast, self.log_w_cross, self.log_w_rec, self.log_w_msc], "lr":0.005})

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12)

        # Logs
        loss_log = []
        acc_log = []
        nmi_log = []
        contr_log = []
        cross_log = []
        rec_log = []

        # Pretraining loop
        for T in range(0, self.pretraining_epoch):
            logger.info(f'Pretraining Epoch: {T + 1}')
            total_loss = 0
            total_contrastive_loss = 0
            total_cross_loss = 0
            total_rec_loss = 0
            num_batches = 0
            for x, target, _ in self.data_loader:
                optimizer.zero_grad()

                x = x.to(self.device)
                sample_size = x.size(0)

                # Encode views
                z_views = self.encode_views(x, use_mask=True)

                # Initialize losses
                contrastive_loss = 0
                cross_loss = 0
                reconstruction_loss = 0

                # Inter-view-Constrastive Loss
                for i in range(self.num_views):
                    for j in range(i + 1, self.num_views):
                        z_i = z_views[i]
                        z_j = z_views[j]
                        
                        # Cross-view reconstruction
                        z_i_recon = self.cross_view_decoders[f'{i}_to_{j}'](z_i)
                        z_j_recon = self.cross_view_decoders[f'{j}_to_{i}'](z_j)
                        cross_loss += F.mse_loss(z_j_recon, z_j.detach()) / sample_size
                        cross_loss += F.mse_loss(z_i_recon, z_i.detach()) / sample_size

                        # Contrastive loss
                        view_contrastive_loss = self.contrastive_loss(z_i_recon, z_j_recon)
                        contrastive_loss += view_contrastive_loss
                contrastive_loss /= (self.num_views * (self.num_views - 1))
                cross_loss /= (self.num_views * (self.num_views - 1))
                

                # Reconstruction Loss
                for i in range(self.num_views):
                    z_i = z_views[i]
                    recon_i = self.view_decoders[f'view_{i}_decoder'](z_i)
                    reconstruction_loss += F.mse_loss(recon_i, x.detach()) / sample_size
                reconstruction_loss /= self.num_views

                # Calcola i pesi ad ogni iterazione (nuovo grafo)
                w_contrast = F.softplus(self.log_w_contrast)
                w_cross = F.softplus(self.log_w_cross)
                w_rec = F.softplus(self.log_w_rec)

                loss = w_contrast * contrastive_loss + w_cross * cross_loss + w_rec * reconstruction_loss
                loss.backward()

                # ---- Gradient Clipping ----
                torch.nn.utils.clip_grad_norm_(
                    list(self.view_encoders.parameters()) +
                    list(self.view_decoders.parameters()) +
                    list(self.cross_view_decoders.parameters()),
                    1.0
                )
                optimizer.step()
                
                # Update SWA model
                self.__dict__['net'].update_parameters(self.encoder_module)

                # Accumulate losses for logging
                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_cross_loss += cross_loss.item()
                total_rec_loss += reconstruction_loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            average_contrastive_loss = total_contrastive_loss / num_batches
            average_cross_loss = total_cross_loss / num_batches
            average_rec_loss = total_rec_loss / num_batches
            scheduler.step(average_loss)

            ACC, NMI = self.Kmeans_model_evaluation(T, logger)
            acc_log.append(ACC)
            nmi_log.append(NMI)
            loss_log.append(average_loss)
            cross_log.append(average_cross_loss)
            rec_log.append(average_rec_loss)
            contr_log.append(average_contrastive_loss)
            logger.info(f"Epoch #{T + 1}: "
                f"loss={average_loss:.4f}, "
                f"contrastive_loss={average_contrastive_loss:.4f}, "
                f"cross_loss={average_cross_loss:.4f}, "
                f"rec_loss={average_rec_loss:.4f}, "
                f"ACC={ACC}, "
                f"NMI={NMI}")

        file = os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name + '/pretraining.csv' if self.mode != 'unidirectional' else os.getcwd() + '/launches/' + self.dataset_name + '/pretraining.csv'
        if not os.path.exists(os.getcwd() + '/launches/' + self.dataset_name) and self.mode == 'unidirectional':
            os.makedirs(os.getcwd() + '/launches/' + self.dataset_name)
        if not os.path.exists(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name) and self.mode != 'unidirectional':
            os.makedirs(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name)
        data = pd.DataFrame.from_dict({'pretraining': loss_log, 'contrastive_loss': contr_log, 'rec_loss': rec_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')
        data.to_csv(file, index=False)
        
        if T == self.pretraining_epoch-1:
            save_path = f'launches_{self.mode}/' + self.dataset_name + '/Pretraining_phase.pt' \
                if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + '/Pretraining_phase.pt'
            torch.save({
                'view_encoders': self.view_encoders.state_dict(),
                'view_decoders': self.view_decoders.state_dict(),
                'cross_view_decoders': self.cross_view_decoders.state_dict(),
            }, save_path)

        return self.net

    def Finetuning(self, logger):
        # Initialiazation
        net_obj, self.u_mean = self.initialization(logger)
        self.__dict__['net'] = net_obj
        
        # Set all parameters to require gradients
        self.view_encoders.train()
        self.view_decoders.train()
        self.cross_view_decoders.train()
        
        for param in self.view_encoders.parameters():
            param.requires_grad = True
        for param in self.view_decoders.parameters():
            param.requires_grad = True
        for param in self.cross_view_decoders.parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW(
            list(self.view_encoders.parameters()) +
            list(self.view_decoders.parameters()) +
            list(self.cross_view_decoders.parameters()),
            lr=0.0001
        )

        optimizer.add_param_group({
            'params': [self.log_w_rec, self.log_w_msc],
            "lr":0.005
        })
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12)

        # Logs
        loss_log = []
        acc_log = []
        nmi_log = []
        rec_log = []
        cluster_log = []
        
        # ---- Finetuning loop ----
        for T in range(0, self.MaxIter1):
            logger.info(f'Finetuning Epoch: {T + 1}')
            total_loss = 0
            total_rec_loss = 0
            total_cluster_loss = 0
            num_batches = 0

            # Update cluster centers every T1 epochs
            if T % self.T1 == 1:
                self.u_mean = self.update_cluster_centers().to(self.device)
            
            for x, label_train, index in self.data_loader:
                x = x.to(self.device)
                sample_size = x.size(0)

                optimizer.zero_grad()

                # Encoding
                z_views = self.encode_views(x, use_mask=True)

                # Reconstruction Loss
                reconstruction_loss = 0
                for i in range(self.num_views):
                    recon = self.view_decoders[f'view_{i}_decoder'](z_views[i])
                    reconstruction_loss += F.mse_loss(recon, x.detach())

                reconstruction_loss /= self.num_views

                # Clustering Loss
                u = self.encode_with_pooling(x)  # [B, D]
                loss_c = self.series_clustering_loss(u)

                loss_c /= self.num_views

                # Learnable weights for losses
                w_rec = F.softplus(self.log_w_rec)
                w_msc = F.softplus(self.log_w_msc)

                loss = (
                    w_rec * reconstruction_loss +
                    w_msc * loss_c
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.view_encoders.parameters()) +
                    list(self.view_decoders.parameters()) +
                    list(self.cross_view_decoders.parameters()),
                    1.0
                )

                optimizer.step()
                
                # Update SWA model
                self.__dict__['net'].update_parameters(self.encoder_module)

                # Accumulate losses for logging
                total_loss += loss.item()
                total_rec_loss += reconstruction_loss.item()
                total_cluster_loss += loss_c.item()
                num_batches += 1
            
            # Average losses over batches
            average_loss = total_loss / num_batches
            average_rec_loss = total_rec_loss / num_batches
            average_cluster_loss = total_cluster_loss / num_batches
            logger.info(f'Finetuning Epoch: {T + 1}, Total Loss: {average_loss}, Rec Loss: {average_rec_loss}, Cluster Loss: {average_cluster_loss}')
            scheduler.step(average_loss)

            ACC, NMI = self.model_evaluation(T, logger)

            acc_log.append(ACC)
            nmi_log.append(NMI)
            loss_log.append(average_loss)
            rec_log.append(average_rec_loss)
            cluster_log.append(average_cluster_loss)

            if T == self.MaxIter1 - 1:
                torch.save(self.state_dict(),
                        f'launches_{self.mode}/' + self.dataset_name + '/Finetuning_phase.pt' if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + '/Finetuning_phase.pt'
                        )

                torch.save(self.u_mean,
                        f'launches_{self.mode}/' + self.dataset_name + '/Centers.pt' if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + '/Centers.pt'
                        )

            logger.info(
                f"Finetuning Epoch: {T + 1}: "
                f"loss={average_loss:.4f}, "
                f"rec_loss={average_rec_loss:.4f},"
                f"cluster_loss={average_cluster_loss:.4f}, "
                f"ACC={ACC}, "
                f"NMI={NMI}"
            )

        file = os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name + '/finetuning.csv' if self.mode != 'unidirectional' else os.getcwd() + '/launches/' + self.dataset_name + '/finetuning.csv'
        if not os.path.exists(os.getcwd() + '/launches/' + self.dataset_name) and self.mode == 'unidirectional':
            os.makedirs(os.getcwd() + '/launches/' + self.dataset_name)
        if not os.path.exists(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name) and self.mode != 'unidirectional':
            os.makedirs(os.getcwd() + f'/launches_{self.mode}/' + self.dataset_name)
        data = pd.DataFrame.from_dict({'finetuning': loss_log, 'rec_loss': rec_log, 'cluster_loss': cluster_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')
        data.to_csv(file, index=False)

    def initialization(self, logger):
        logger.info("-----initialization mode--------")

        # Load pretraining weights
        pretrain_path = f'launches_{self.mode}/' + self.dataset_name + '/Pretraining_phase.pt' \
            if self.mode != 'unidirectional' else 'launches/' + self.dataset_name + '/Pretraining_phase.pt'
        
        state_dict = torch.load(pretrain_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

        self.encoder_module = MultiViewEncoder(
            self.view_encoders,
            self.view_decoders,
            self.cross_view_decoders
        )
        self.__dict__['net'] = torch.optim.swa_utils.AveragedModel(self.encoder_module)
        self.__dict__['net'].update_parameters(self.encoder_module)
        
        # Code data for KMeans
        datas = torch.zeros(
            self.dataset_size,
            self.latent_size,
            device=self.device
        )
        ii = 0
        for x, _, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u
            ii += 1

        datas_np = datas.cpu().numpy().astype(np.float32)
        kmeans = faiss.Kmeans(
            d=self.latent_size,
            k=self.num_cluster,
            niter=30,
            gpu=True
        )

        kmeans.train(datas_np)
        self.u_mean = torch.from_numpy(kmeans.centroids).to(self.device)
        
        return self.__dict__['net'], self.u_mean
    
    def series_clustering_loss(self, u):

        K = self.num_cluster

        if not hasattr(self, "series_centers"):
            self.series_centers = torch.nn.Parameter(
                torch.randn(K, u.shape[1], device=u.device)
            )

        u_norm = F.normalize(u, dim=1)
        c_norm = F.normalize(self.series_centers, dim=1)

        sim = torch.matmul(u_norm, c_norm.T)

        q = torch.softmax(sim / 0.5, dim=1)

        # entropy
        entropy = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
        entropy = entropy / math.log(K)

        # balance
        p = q.mean(dim=0)
        balance = torch.sum(p * torch.log(p + 1e-8))
        balance = balance / math.log(K)

        # center separation
        center_sim = torch.matmul(c_norm, c_norm.T)

        mask = torch.eye(K, device=u.device).bool()
        center_sim = center_sim.masked_fill(mask, 0)

        separation = torch.mean(center_sim**2)

        loss = entropy + 0.2 * balance + 0.5 * separation

        return loss

    def encode_views(self, x, use_mask=True):
        if use_mask:
            views = MASK(
                x,
                missing_rate=0.3,
                num_view=self.num_views
            )
        else:
            views = [x for _ in range(self.num_views)]

        latents = []
        for i, v in enumerate(views):
            encoder = self.view_encoders[f'view_{i}']
            z = encoder(v.to(self.device))
            latents.append(z)

        return latents
    
    def Kmeans_model_evaluation(self, T, logger):
        self.view_encoders.eval()

        datas_list = []
        label_true_list = []
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            if u is None:
                raise ValueError("encode_with_pooling(x) returned None")
            datas_list.append(u)
            label_true_list.append(target.numpy())

        datas = torch.cat(datas_list, dim=0)
        if datas.numel() == 0:
            raise ValueError("Empty datas tensor")
        label_true = np.concatenate(label_true_list, axis=0)
        datas_np = datas.detach().cpu().numpy().astype(np.float32)

        # FAISS KMeans on GPU
        kmeans = faiss.Kmeans(
            d=self.latent_size,
            k=self.num_cluster,
            niter=30,
            gpu=True
        )

        kmeans.train(datas_np)
        self.u_mean = torch.from_numpy(kmeans.centroids).to(self.device)

        # Assign clusters
        _, labels_pred = kmeans.index.search(datas_np, 1)
        labels_pred = labels_pred.ravel().astype(int)

        # Check length
        assert labels_pred.size == label_true.size, f"labels_pred ({labels_pred.size}) != label_true ({label_true.size})"

        ACC = acc(label_true, labels_pred, self.num_cluster)
        NMI = nmi(label_true, labels_pred)
        logger.info(f'ACC: {ACC}')
        logger.info(f'NMI: {NMI}')

        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        os.makedirs(f'./{name}/features', exist_ok=True) 
        if T == 0:
            np.save(f'./{name}/features/Start_Pretraining_R.npy', datas_np)
            np.save(f'./{name}/features/Start_Pretraining_y_true.npy', label_true)
        if T == self.pretraining_epoch-1:
            np.save(f'./{name}/features/End_Pretraining_R.npy', datas_np)
            np.save(f'./{name}/features/End_Pretraining_y_true.npy', label_true)

        self.view_encoders.train()
        return ACC, NMI
    
    def calculate_cluster_loss(self, fused_data, labels):

        sim = F.cosine_similarity(
            fused_data.unsqueeze(1),
            fused_data.unsqueeze(0),
            dim=2
        )

        loss = torch.tensor(0.0, device=fused_data.device)
        N = fused_data.size(0)

        for i in range(N):

            pos_mask = labels == labels[i]
            pos_mask[i] = False

            neg_mask = ~pos_mask

            pos_sim = sim[i][pos_mask]
            neg_sim = sim[i][neg_mask]

            if len(pos_sim) == 0:
                continue

            logits = torch.cat([pos_sim, neg_sim])
            targets = torch.zeros(len(pos_sim), dtype=torch.long,
                                device=fused_data.device)

            loss += F.cross_entropy(
                logits.unsqueeze(0).repeat(len(pos_sim), 1),
                targets
            )

        return loss / max(N, 1)

    def update_cluster_centers(self):
        self.view_encoders.eval()
        for param in self.view_encoders.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).to(self.device)
        num = torch.zeros([self.num_cluster, self.latent_size]).to(self.device)

        for x, _, _ in self.data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)

            p = self.cmp(u.unsqueeze(0).repeat(self.num_cluster, 1, 1), self.u_mean)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].T, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])

        self.u_mean = F.normalize(self.u_mean, dim=1)
        self.view_encoders.train()
        # Unfreeze parameters
        for param in self.view_encoders.parameters():
            param.requires_grad = True
        return self.u_mean

    def cmp(self, u, u_mean):
        real_batch_size = u.size(1)
        p = torch.zeros([real_batch_size, self.num_cluster]).to(self.device)
        for j in range(0, self.num_cluster):
            p[:, j] = torch.sum(torch.pow(u[j, :, :] - u_mean[j, :].unsqueeze(0).repeat(real_batch_size, 1), 2), dim=1)
        p = torch.pow(p, -1 / (self.m - 1))
        sum1 = torch.sum(p, dim=1)
        p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
        return p

    def model_evaluation(self, T, logger):
        datas = np.zeros([self.dataset_size, self.latent_size])
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u.data.cpu().numpy()

            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            pred_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = y
            true_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = target.numpy()
            ii = ii + 1

        ACC = acc(true_labels, pred_labels, self.num_cluster)
        NMI = nmi(true_labels, pred_labels)
        logger.info(f'ACC: {ACC}')
        logger.info(f'NMI: {NMI}')
        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        os.makedirs(f'./{name}/features', exist_ok=True)
        if T == 0:
            np.save(f'./{name}/features/Start_Finetuning_R.npy', datas)
            np.save(f'./{name}/features/Start_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./{name}/features/Start_Finetuning_y_true.npy', true_labels)
        if T == self.MaxIter1-1:
            np.save(f'./{name}/features/End_Finetuning_End_Finetuning_R.npy', datas)
            np.save(f'./{name}/features/End_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./{name}/features/End_Finetuning_y_true.npy', true_labels)
        self.view_encoders.train()
        for param in self.view_encoders.parameters():
            param.requires_grad = True

        return ACC, NMI

    def encode_with_pooling(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        assert x.ndim == 3

        was_training_encoders = self.view_encoders.training
        self.view_encoders.eval()
        z_views = self.encode_views(x, use_mask=False)

        pooled_views = []
        for z in z_views:
            B, T, D = z.shape

            centers = F.normalize(self.state_centers, dim=1)
            z_norm = F.normalize(z, dim=2)

            sim = torch.matmul(z_norm, centers.T)   # [B,T,K]
            weights = torch.softmax(sim.mean(dim=2), dim=1).unsqueeze(-1)
            pooled = torch.sum(z * weights, dim=1)

            pooled_views.append(pooled)

        fused = torch.stack(pooled_views, dim=0).mean(dim=0)
        fused = F.normalize(fused, dim=1)

        if was_training_encoders:
            self.view_encoders.train()
            
        return fused
    
    def mask_instance_loss_with_mixup(self, z1, z2, pseudo_label=None):
        # Evaluation of the loss function with mixup
        B, T = z1.size(0), z1.size(1)
        temp = 1.0

        # If no pseudo label is provided, set it to -1
        if pseudo_label == None:
            pseudo_label = torch.full((B,), -1, dtype=torch.int64).to(self.device)

        # If batch size is 1, return 0 (needs at least 2 samples)
        if B == 1:
            return z1.new_tensor(0.)

        pseudo_label = pseudo_label.to(z1.device)

        # Hard weight
        hard_w = self.hard_w

        # Generate hard positive and negative samples and evaluate h1
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z1 = hard_w * z1[pos_indices, :, :] + (1 - hard_w) * z1[neg_indices, :, :].view(z1.size())

        # Generate hard positive and negative samples and evaluate h2
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z2 = hard_w * z2[pos_indices, :, :] + (1 - hard_w) * z2[neg_indices, :, :].view(z2.size())

        # Concatenate the original and hard samples
        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)

        # Transpose the matrix (loss evaluated per timestep)
        z = z.transpose(0, 1) 
        
        # Similarity matrix (dot product) --> preparing the denominator of the loss function
        sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2))

        # Invalid index
        invalid_index = pseudo_label == -1

        # Mask cluster-aware
        mask = torch.eq(
            pseudo_label.view(-1, 1),
            pseudo_label.view(1, -1)
        ).to(z1.device)

        # Invalid index
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False

        # Mask out self-contrast
        mask_eye = torch.eye(B).float().to(z1.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        # Adapting to sim shape
        mask = mask.repeat(2, 4)
        mask_eye = mask_eye.repeat(2, 4)

        # Initializing logits mask
        logits_mask = torch.ones(2 * B, 4 * B).to(z1.device)

        # Deleting self-contrast
        rows = torch.arange(2 * B).view(-1, 1).to(z1.device)
        logits_mask = logits_mask.scatter(1, rows, 0)

        # Deleting positive samples (for denominator)
        logits_mask *= 1 - mask

        # Building self-positive samples (for numerator)
        mask_eye = mask_eye * logits_mask

        # Numeric Stabilization (avoiding overflow for exp)
        logits = sim
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        logits = logits - logits_max

        # It's the full denominator of the loss function
        neg_exp_logits = torch.exp(logits) * logits_mask
        neg_exp_log_sum = neg_exp_logits.sum(-1, keepdim=True)

        # It's the full numerator of the loss function
        pos_exp_log = torch.exp(logits)

        # Total loss
        prob = pos_exp_log / (neg_exp_log_sum + 1e-10)

        # Selection of the positive samples (instance-level positive base)
        prob = prob[:, 0:B, B:2 * B]

        # Building final positive mask
        mask = mask[:B, : B]
        self_mask = mask_eye[:B, B:2 * B]
        diffaug_cluster_mask = mask
        pos_mask = (self_mask + diffaug_cluster_mask)

        # Final positive probability sum
        pos_prob_sum = (prob * pos_mask.unsqueeze(0)).sum(-1)

        # Log probability
        log_prob = torch.log(pos_prob_sum + 1e-10)

        # Final log probability (mean over timesteps)
        log_prob = log_prob.sum(dim=0) / T

        # Final loss
        instance_loss = -log_prob
        instance_loss = instance_loss.mean()

        return instance_loss

    def contrastive_loss(
        self,
        z1, 
        z2, 
        mask=False,
        pseudo_label=None,
        alpha=0.8,
    ):
        """
        Instance CL + Mamba-aware temporal CL
        """

        instance_loss = torch.tensor(0., device=z1.device)

        # -------- Instance-level CL --------
        if alpha > 0:
            if not mask:
                instance_loss += self.mask_instance_loss_with_mixup(z1, z2)
            else:
                instance_loss += self.mask_instance_loss_with_mixup(z1, z2, pseudo_label)

        return instance_loss

    def pooling(self, x_whole_list):

        pooled_list = []

        for x_view in x_whole_list:

            x_view = x_view.to(self.device)
            num_samples, seq_len, feature_dims = x_view.shape

            var_per_sample = torch.var(x_view, dim=2).sum(dim=1)  # (num_samples,)
            var_min = var_per_sample.min()
            var_max = var_per_sample.max()
            time_steps = torch.arange(seq_len, device=self.device).float()

            pooled_samples = []
            for i in range(num_samples):

                alpha = (var_per_sample[i] - var_min) / (var_max - var_min + 1e-8)


                logits = alpha * time_steps
                weights = torch.softmax(logits, dim=0)


                pooled = torch.sum(x_view[i] * weights.view(-1, 1), dim=0)
                pooled_samples.append(pooled)

            pooled_list.append(torch.stack(pooled_samples).cpu().numpy())

        return pooled_list
    
    # Define the function to evaluate real test data
    def eval_with_test_data(self, dataset_name, logger, data_loader, save=False):
        self.view_encoders.eval()

        data = np.zeros([self.dataset_size, self.timesteps_len, self.input_dims])
        reps = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        label_pred = np.zeros(self.dataset_size)

        ii = 0
        for x, target, _ in data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)

            reps[ii * self.batch_size: ii * self.batch_size + real_batch_size, :] = u.data.cpu().numpy()
            data[ii * self.batch_size: ii * self.batch_size + real_batch_size, :, :] = x.cpu().numpy()

            # Get predicted labels
            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu().numpy()

            label_true[ii * self.batch_size: ii * self.batch_size + real_batch_size] = target.numpy()
            label_pred[ii * self.batch_size: ii * self.batch_size + real_batch_size] = y

            ii = ii + 1

        # Evaluate performance
        logger.info("-------testdata_Evaluate---------")

        name = 'results' if self.mode == 'unidirectional' else f'results_{self.mode}'
        save_dir = f'./{name}/{self.dataset_name}/label/'
        os.makedirs(save_dir, exist_ok=True)

        df_true = pd.DataFrame(label_true, columns=['label_true'])
        df_true.to_csv(f'{save_dir}/{dataset_name}_label_true.csv', index=False)

        df_pred = pd.DataFrame(label_pred, columns=['label_pred'])
        df_pred.to_csv(f'{save_dir}/{dataset_name}_label_pred.csv', index=False)
        
        accuracy = acc(label_true, label_pred, self.num_cluster)
        nmi_score = nmi(label_true, label_pred)
        ari_score = ari(label_true, label_pred)
        fmi_score = fmi(label_true, label_pred)
        test_ri = rand_index_score(label_pred, label_true)
        f1_score = f1(label_true, label_pred, average='macro')

        self.view_encoders.train()
        return accuracy, nmi_score, ari_score, test_ri, fmi_score, f1_score