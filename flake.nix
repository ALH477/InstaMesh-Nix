{
  description = "Production-ready Nix flake for InstantMesh with NVIDIA CUDA and AMD ROCm support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        rocmSupport = true;
      };
    };

    python = pkgs.python310;

    # CUDA Dockerfile – closely based on the official one but self-contained (git clone inside)
    cudaDockerfileText = ''
      FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

      LABEL name="instantmesh-cuda" maintainer="DeMoD (@DeMoDLLC)"

      VOLUME /workspace/models

      WORKDIR /workspace

      ENV DEBIAN_FRONTEND=noninteractive

      RUN apt-get update && \
          apt-get install -y git wget vim libegl1-mesa-dev libglib2.0-0 unzip build-essential && \
          rm -rf /var/lib/apt/lists/*

      # Install Miniconda (matches official setup)
      RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
          chmod +x miniconda.sh && \
          ./miniconda.sh -b -p /workspace/miniconda3 && \
          rm miniconda.sh

      ENV PATH="/workspace/miniconda3/bin:${PATH}"

      RUN conda init bash

      RUN conda create -n instantmesh python=3.10 -y && \
          conda clean -afy

      ENV PATH="/workspace/miniconda3/envs/instantmesh/bin:${PATH}"
      ENV CONDA_DEFAULT_ENV=instantmesh

      RUN conda install ninja -y && conda clean -afy
      RUN conda install cuda -c nvidia/label/cuda-12.4.1 -y && conda clean -afy

      RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
      RUN pip install --no-cache-dir xformers==0.0.22.post7 triton

      # Clone repository
      RUN git clone https://github.com/TencentARC/InstantMesh.git /workspace/instantmesh

      WORKDIR /workspace/instantmesh

      RUN pip install --no-cache-dir -r requirements.txt

      EXPOSE 43839  # Port used in official Docker instructions

      CMD ["python", "app.py"]
    '';

    # ROCm Dockerfile – experimental, pip-based (no official ROCm support)
    rocmDockerfileText = ''
      FROM rocm/dev-ubuntu-22.04:6.1.0-complete

      LABEL name="instantmesh-rocm" maintainer="DeMoD (@DeMoDLLC)"

      VOLUME /workspace/models

      WORKDIR /workspace

      ENV DEBIAN_FRONTEND=noninteractive

      RUN apt-get update && \
          apt-get install -y git python3 python3-pip python3-venv ninja-build libglib2.0-0 libgl1-mesa-glx && \
          rm -rf /var/lib/apt/lists/*

      # Clone repository
      RUN git clone https://github.com/TencentARC/InstantMesh.git /workspace/instantmesh

      WORKDIR /workspace/instantmesh

      RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/

      # xformers ROCm support is limited – fallback to source build if wheel unavailable
      RUN pip install --no-cache-dir xformers || (pip install --no-cache-dir git+https://github.com/facebookresearch/xformers.git)

      RUN pip install --no-cache-dir -r requirements.txt

      # Warning: Some dependencies (e.g., nvdiffrast) are NVIDIA-specific and may fail or have reduced functionality on ROCm

      EXPOSE 43839

      CMD ["python", "app.py"]
    '';

    mkDevShell = backend: pkgs.mkShell {
      name = "instantmesh-${backend}";

      packages = [ python pkgs.git pkgs.ninja ];

      buildInputs = if backend == "cuda" then
        (with pkgs; [ cudatoolkit cudaPackages.cudnn linuxPackages.nvidia_x11 ])
      else
        (with pkgs.rocmPackages; [ clr rocm-runtime miopen hipblas rccl ]);

      shellHook = ''
        if [ ! -d "InstantMesh" ]; then
          echo "Cloning InstantMesh repository..."
          git clone https://github.com/TencentARC/InstantMesh.git InstantMesh
        fi

        cd InstantMesh

        if [ ! -d ".venv" ]; then
          echo "Creating virtual environment and installing dependencies (this may take a while)..."
          ${python}/bin/python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip

          ${if backend == "cuda" then ''
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            pip install xformers==0.0.22.post7
          '' else ''
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/
            pip install xformers || (echo "Building xformers from source..." && pip install git+https://github.com/facebookresearch/xformers.git)
          ''}

          pip install -r requirements.txt
          deactivate
        fi

        source .venv/bin/activate

        echo "InstantMesh ${backend} environment activated."
        echo " - Run the Gradio demo: python app.py"
        echo " - Run CLI inference: python run.py configs/instant-mesh-large.yaml <image.png> [options]"
      '';

      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (if backend == "cuda" then
        [ pkgs.cudatoolkit pkgs.linuxPackages.nvidia_x11 ]
      else
        [ pkgs.rocmPackages.rocm-runtime ]);
    };

  in {
    devShells = {
      cuda = mkDevShell "cuda";
      rocm = mkDevShell "rocm";
    };

    packages = {
      cuda-dockerfile = pkgs.writeText "Dockerfile.cuda" cudaDockerfileText;
      rocm-dockerfile = pkgs.writeText "Dockerfile.rocm" rocmDockerfileText;
    };
  });
}
