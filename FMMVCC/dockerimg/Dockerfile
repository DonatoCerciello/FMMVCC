FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG UID=$UID$
ARG GID=$GID$
ARG USER=$USER$
ARG PASSWORD=$PASSWORD$

ENV DEBIAN_FRONTEND=noninteractive
ENV PASSWORD=${PASSWORD}

# ---- SYSTEM SETUP ----
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    curl \
    htop \
    nvtop \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- USER SETUP ----
RUN groupadd --gid $GID $USER && \
    useradd -m -u $UID -g $GID -s /bin/bash $USER

WORKDIR /home/$USER/PhD

# ---- PYTHON DEPENDENCIES ----
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    matplotlib \
    scipy \
    pandas \
    scikit-learn \
    tslearn \
    umap-learn \
    faiss-gpu-cu12 \
    jupyter \
    jupyterlab

RUN MAX_JOBS=$(nproc) pip3 install --no-cache-dir --break-system-packages --no-build-isolation \
    "mamba-ssm[causal-conv1d]"

# ---- SWITCH USER ----
USER $USER

# ---- JUPYTER CONFIG ----
RUN jupyter lab --generate-config && \
    python3 -c "from jupyter_server.auth import passwd; \
    print(f\"c.ServerApp.password='{passwd('${PASSWORD}')}'\")" \
    >> /home/$USER/.jupyter/jupyter_lab_config.py

# ---- EXPOSE PORT ----
EXPOSE 8888

# ---- START ----
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]