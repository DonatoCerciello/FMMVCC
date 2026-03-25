import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import json
import torch
from collections import Counter
from pathlib import Path

PATH = Path(__file__).parent.absolute()

def plot_latent_space(u1, y_train, model_name, save_title="", plot_root=Path(PATH / 'plot')):
    # -----------------------------
    # Standardizzazione
    # -----------------------------
    u1_scaled = StandardScaler().fit_transform(u1)

    # -----------------------------
    # PCA 2D
    # -----------------------------
    pca = PCA(n_components=2)
    u1_pca = pca.fit_transform(u1_scaled)

    perplexity = min(30, len(u1_scaled) - 1)  # t-SNE perplexity < number of samples
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )
    u1_tsne = tsne.fit_transform(u1_scaled)

    # -----------------------------
    # UMAP 2D
    # -----------------------------
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    u1_umap = umap_model.fit_transform(u1_scaled)

    # -----------------------------
    # Color map discreta
    # -----------------------------
    classes = np.unique(y_train)
    cmap = plt.get_cmap("viridis", len(classes))  
    norm = matplotlib.colors.BoundaryNorm(boundaries=np.arange(len(classes)+1)-0.5, ncolors=len(classes))

    # -----------------------------
    # Plot: PCA vs t-SNE vs UMAP
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA
    sc0 = axes[0].scatter(u1_pca[:, 0], u1_pca[:, 1], c=y_train, cmap=cmap, norm=norm, s=40)
    axes[0].set_title("PCA 2D")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # t-SNE
    sc1 = axes[1].scatter(u1_tsne[:, 0], u1_tsne[:, 1], c=y_train, cmap=cmap, norm=norm, s=40)
    axes[1].set_title("t-SNE 2D")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    # UMAP
    sc2 = axes[2].scatter(u1_umap[:, 0], u1_umap[:, 1], c=y_train, cmap=cmap, norm=norm, s=40)
    axes[2].set_title("UMAP 2D")
    axes[2].set_xlabel("UMAP 1")
    axes[2].set_ylabel("UMAP 2")

    # Labels
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc2, cax=cax, ticks=classes)
    cbar.set_label("Class")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    # Salva la figura
    if not os.path.exists(os.path.join(plot_root, model_name)):
        os.makedirs(os.path.join(plot_root, model_name))
    fig_path = os.path.join(plot_root, model_name, f"{save_title}_Latent_Space_PCA_tSNE_UMAP.png")
    fig.savefig(fig_path, bbox_inches='tight', format='png')
    plt.close(fig)

def build_label_mapping(y_true, y_pred):
    mapping = {}
    for pred_label in np.unique(y_pred):
        true_vals = y_true[y_pred == pred_label]
        best_true = Counter(true_vals).most_common(1)[0][0]
        mapping[pred_label] = best_true
    return mapping

def apply_mapping(y_pred, mapping_train, mapping_test=None):
    aligned_pred = []
    unique_labels = set(mapping_train.values()) 
    
    for v in y_pred:
        if v in mapping_train:
            aligned_pred.append(mapping_train[v])
        elif mapping_test and v in mapping_test:
            aligned_pred.append(mapping_test[v])
        else:
            new_label = f"new_{len(unique_labels)}"  
            aligned_pred.append(new_label)
            unique_labels.add(new_label) 
    
    return np.array(aligned_pred)

def update_dataset_registry(
    json_path: Path,
    dataset_name: str,
    position: int,
    univariate: bool,
    n_clusters: int,
    train_shape: int,
    temporal_length: int
    ):
    """
    Update a JSON registry of processed datasets.
    
    Structure:
    {
        "univariate": {dataset_name: position, ...},
        "multivariate": {dataset_name: position, ...}
    }
    """

    key = "univariate" if univariate else "multivariate"

    # Initialize empty structure if file does not exist
    if json_path.exists():
        with open(json_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"univariate": {}, "multivariate": {}}

    # Safety check
    if key not in registry:
        registry[key] = {}

    # Add dataset only if not present
    if dataset_name not in registry[key]:
        registry[key][dataset_name] = {
            "position": position,
            "n_clusters": n_clusters,
            "train_shape": train_shape,
            "temporal_length": temporal_length
        }

        # Sort datasets alphabetically
        registry[key] = dict(sorted(registry[key].items()))

        # Save back to file
        with open(json_path, "w") as f:
            json.dump(registry, f, indent=4)

        return True  # Added
    else:
        return False  # Already present

def estimate_seasonality_generic(X, max_period=None):
    """
    X: np.array shape (N, T, 1)
    Ritorna: periodo stimato o None
    """

    X = np.asarray(X)
    N, T, _ = X.shape

    if max_period is None:
        max_period = T // 2

    # FFT sulla media delle serie
    mean_ts = X[:, :, 0].mean(axis=0)
    mean_ts -= mean_ts.mean()

    fft = np.abs(np.fft.rfft(mean_ts))
    freqs = np.fft.rfftfreq(T, d=1)

    fft, freqs = fft[1:], freqs[1:]  # rimuovo DC
    periods = 1 / freqs

    valid = (periods > 1) & (periods <= max_period)

    if not np.any(valid):
        return None

    fft_period = int(round(periods[valid][np.argmax(fft[valid])]))

    return fft_period

def plot_mean_series_with_period(X, period):
    """
    X: np.array shape (N, T, 1)
    period: periodo stimato
    """
    X = np.asarray(X)
    mean_ts = X[:, :, 0].mean(axis=0)

    plt.figure(figsize=(15, 5))
    plt.plot(mean_ts, label='Mean series', color='blue')
    
    # linee verticali ad ogni periodo
    for i in range(period, len(mean_ts), period):
        plt.axvline(i, color='red', linestyle='--', alpha=0.5)

    plt.title(f'Mean Series with Estimated Period = {period}', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def encode_in_batches(model, X, batch_size=64):
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size]).float().to(model.device)
        with torch.no_grad():
            z = model.encode_with_pooling(batch).detach().cpu()
        embeddings.append(z)
    return torch.cat(embeddings).numpy()