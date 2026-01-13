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

    # CUDA Dockerfile – closely matches the official InstantMesh Dockerfile but self-contained (git clone inside)
    # Fixed Nix string interpolation for ${PATH}
    cudaDockerfileText = ''
      FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

      LABEL name="instantmesh-cuda" maintainer="DeMoD (@DeMoDLLC)"

      VOLUME /workspace/models

      WORKDIR /workspace

      ENV DEBIAN_FRONTEND=noninteractive

      RUN apt-get update && \
          apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip && \
          rm -rf /var/lib/apt/lists/*

      # Install Miniconda
      RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
          chmod +x miniconda.sh && \
          ./miniconda.sh -b -p /workspace/miniconda3 && \
          rm miniconda.sh

      ENV PATH="/workspace/miniconda3/bin:''${PATH}"

      RUN conda init bash

      RUN conda create -n instantmesh python=3.10 -y && \
          conda clean -afy

      ENV PATH="/workspace/miniconda3/envs/instantmesh/bin:$PATH"

      RUN conda install ninja -y && conda clean -afy
      RUN conda install cuda -c nvidia/label/cuda-12.4.1 -y && conda clean -afy

      RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
      RUN pip install --no-cache-dir xformers==0.0.22.post7 triton

      # Clone repository (self-contained – no need for local context COPY)
      RUN git clone https://github.com/TencentARC/InstantMesh.git /workspace/instantmesh

      WORKDIR /workspace/instantmesh

      RUN pip install --no-cache-dir -r requirements.txt

      EXPOSE 7860  # Gradio default port

      CMD ["python", "app.py"]
    '';

    # ROCm Dockerfile – experimental (no official ROCm support)
    rocmDockerfileText = ''
      FROM rocm/dev-ubuntu-22.04:6.1.0-complete

      LABEL name="instantmesh-rocm" maintainer="DeMoD (@DeMoDLLC)"

      VOLUME /workspace/models

      WORKDIR /workspace

      ENV DEBIAN_FRONTEND=noninteractive

      RUN apt-get update && \
          apt-get install -y git python3 python3-pip python3-venv ninja-build libglib2.0-0 libgl1-mesa-glx && \
          rm -rf /var/lib/apt/lists/*

      # Create virtual environment
      RUN python3 -m venv /workspace/venv
      ENV PATH="/workspace/venv/bin:$PATH"

      # Install PyTorch for ROCm 6.1 (use compatible version; newer PyTorch recommended for ROCm 6.1)
      RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/

      # xformers ROCm support limited – fallback to source if needed
      RUN pip install --no-cache-dir xformers || pip install --no-cache-dir git+https://github.com/facebookresearch/xformers.git

      # Clone repository
      RUN git clone https://github.com/TencentARC/InstantMesh.git /workspace/instantmesh

      WORKDIR /workspace/instantmesh

      RUN pip install --no-cache-dir -r requirements.txt

      # Note: NVIDIA-specific components like nvdiffrast may fail or have reduced functionality on ROCm

      EXPOSE 7860

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
          echo "Creating virtual environment and installing dependencies..."
          ${python}/bin/python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip

          ${if backend == "cuda" then ''
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            pip install xformers==0.0.22.post7 triton
          '' else ''
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/
            pip install xformers || pip install git+https://github.com/facebookresearch/xformers.git
          ''}

          pip install -r requirements.txt
          deactivate
        fi

        source .venv/bin/activate

        echo "InstantMesh ${backend} environment ready!"
        echo " - Gradio demo: python app.py (open http://127.0.0.1:7860)"
        echo " - CLI: python run.py configs/instant-mesh-large.yaml <image.png> [options]"
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
      instantmesh-cuda-dockerfile = pkgs.writeText "Dockerfile.cuda" cudaDockerfileText;
      instantmesh-rocm-dockerfile = pkgs.writeText "Dockerfile.rocm" rocmDockerfileText;
    };
  });
}
