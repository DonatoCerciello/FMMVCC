# FMMVCC: Fuzzy Mamba-based Multi-View Contrastive Clustering for Univariate Time Series

## 📝 Abstract

In many realistic scenarios, large volumes of time series data are generated with limited or expensive annotations. This limitation makes supervised learning methods difficult to apply and leads to the use of unsupervised approaches capable of discovering meaningful structures directly from raw data. Clustering therefore plays a crucial role in organizing time series into groups that share similar temporal patterns, enabling exploratory analysis and downstream tasks without requiring manual labeling. However, existing deep clustering methods often struggle to capture long-range temporal dependencies or rely on architectures with high computational cost. This paper introduces FMMVCC, a Mamba-based deep clustering framework for time series that leverages state space sequence modeling to efficiently learn temporal representations. The model leverages state space sequence modeling to learn temporal representations with linear complexity. Additionally, it utilizes multi-view self-supervised learning with temporal masking and augmentations. An experimental evaluation of the proposed model is performed using existing benchmark datasets.

## 🧠 Framework Overview

![FMMVCC Framework](images/ClusteringFramework.png)

## 🐳 Docker Setup

This repository includes a ready-to-use Docker setup for running the code with GPU support and JupyterLab.

### ✅ 1. Prerequisites

- Docker Engine + Docker Compose
- NVIDIA GPU drivers installed on the host
- NVIDIA Container Toolkit configured for Docker

Optional but recommended on Windows:
- Docker Desktop with WSL2 backend enabled

### ⚙️ 2. Configure Environment Variables

The project uses a `.env` file already present in the repository:

- `UID`: Linux user id in container
- `GID`: Linux group id in container
- `USER`: username created in container
- `PASSWORD`: Jupyter password
- `PORT`: host port mapped to container port 8888

Edit them in `.env`  and in `dockerimg/Dockerfile` before building/running.

### 🏗️ 3. Build the Docker Image

From the project root:

```bash
docker compose build
```

### ▶️ 4. Start the Container

```bash
docker compose up -d
```

### 🌐 5. Open JupyterLab

Open your browser at:

- `http://localhost:<PORT>`

Replace `<PORT>` with the value in `.env` (default: `8866`).
Use the password from `.env` when prompted.

### ⏹️ 6. Stop the Container

```bash
docker compose down
```

## 🚀 Run main.py

You can run the training/evaluation script from inside the container.

1. Run the script from the project root:

```bash
python main.py
```

2. Optional example with explicit arguments:

```bash
python main.py --dataset-name SyntheticControl --batch-size 32 --output-dims 64 --lr 0.001
```

## 📦 mamba-ssm Compatibility Note

For installation details and supported versions, refer to the official PyPI package page:

- https://pypi.org/project/mamba-ssm/


## 🗂️ Data and Resources

The data used in this study are available in the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

## 💎 Acknowledgment

This work was supported by the MSCA Doctoral Networks project T.U.A.I. - Towards an Understanding of Artificial Intelligence via a transparent, open and explainable perspective (TUAI) project, N°101168344.
