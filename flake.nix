{
  description = "Nix flake for InstantMesh with NVIDIA/ROCm support and Docker image building";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    dream2nix.url = "github:nix-community/dream2nix";
  };

  outputs = { self, nixpkgs, flake-utils, dream2nix }: flake-utils.lib.eachDefaultSystem (system: let
    baseConfig = { inherit system; config.allowUnfree = true; };

    pkgsCuda = import nixpkgs (baseConfig // { config.cudaSupport = true; });
    pkgsRocm = import nixpkgs (baseConfig // { config.rocmSupport = true; });

    commonDeps = pkgs: with pkgs; [
      git
      ninja
    ];

    mkInstantMeshPackage = { pkgs, backend }: dream2nix.lib.evalModules {
      packageSets.nixpkgs = pkgs;
      modules = [
        {
          imports = [ dream2nix.modules.dream2nix.pip ];

          deps = { nixpkgs, ... }: {
            python = nixpkgs.python310;
            inherit (nixpkgs) ninja;
          };

          name = "instantmesh";
          version = "0.1.0";

          mkDerivation = {
            src = pkgs.fetchFromGitHub {
              owner = "TencentARC";
              repo = "InstantMesh";
              rev = "main";
              hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Replace with actual hash (run `nix build` and copy from error)
            };
            nativeBuildInputs = [ config.deps.ninja ];
            postInstall = ''
              mkdir -p $out/share/InstantMesh
              cp -r $src/* $out/share/InstantMesh
              mkdir -p $out/bin
              ln -s $out/share/InstantMesh/run.py $out/bin/instantmesh-run
              ln -s $out/share/InstantMesh/app.py $out/bin/instantmesh-app
              wrapProgram $out/bin/instantmesh-run --prefix PYTHONPATH : "$PYTHONPATH"
              wrapProgram $out/bin/instantmesh-app --prefix PYTHONPATH : "$PYTHONPATH"
            '';
          };

          buildPythonPackage = {
            format = "other";
          };

          pip = {
            requirementsList = [
              "torch==2.1.0"
              "torchvision==0.16.0"
              "torchaudio==2.1.0"
              "xformers==0.0.22.post7"
              "pytorch-lightning==2.1.2"
              "gradio==3.41.2"
              "huggingface-hub"
              "einops"
              "omegaconf"
              "torchmetrics"
              "webdataset"
              "accelerate"
              "tensorboard"
              "PyMCubes"
              "trimesh"
              "rembg[gpu]"
              "transformers==4.34.1"
              "diffusers==0.20.2"
              "bitsandbytes"
              "imageio[ffmpeg]"
              "xatlas"
              "plyfile"
              "git+https://github.com/NVlabs/nvdiffrast/"
            ];
            pipFlags = if backend == "cuda" then [ "--extra-index-url https://download.pytorch.org/whl/cu121" ] else [ "--extra-index-url https://download.pytorch.org/whl/rocm6.0/" ];
            flattenDependencies = true;
          };
        }
        {
          paths.projectRoot = self;
          paths.projectRootFile = "flake.nix";
          paths.package = self;
        }
      ];
    };

    # Docker base images
    cudaBase = pkgsCuda.dockerTools.pullImage {
      imageName = "nvidia/cuda";
      imageTag = "12.1.0-runtime-ubuntu22.04";
      imageDigest = "sha256:a7d949f17fcdfcd56763aa66a7b91c17775a0e1e9d268421b53be3ae2d0e55af";
      sha256 = pkgsCuda.lib.fakeSha256; # Run `nix build .#instantmesh-cuda-docker` and replace with expected hash from error
      os = "linux";
      arch = "x86_64";
    };

    rocmBase = pkgsRocm.dockerTools.pullImage {
      imageName = "rocm/dev-ubuntu-22.04";
      imageTag = "6.0.2";
      imageDigest = "sha256:794671fee7b8cb2cb3046235e834674f6432e242ce6e2e9701f1509d52a62646e";
      sha256 = pkgsRocm.lib.fakeSha256; # Run `nix build .#instantmesh-rocm-docker` and replace with expected hash from error
      os = "linux";
      arch = "x86_64";
    };

    mkInstantMeshDocker = { pkgs, base, backend, package }: pkgs.dockerTools.buildLayeredImage {
      name = "instantmesh-${backend}";
      tag = "latest";
      fromImage = base;
      contents = [ package pkgs.bash pkgs.coreutils ] ++ commonDeps pkgs;
      fakeRootCommands = ''
        mkdir -p /tmp
        chmod 1777 /tmp
      '';
      config = {
        Cmd = [ "/bin/bash" ];
        WorkingDir = "/";
        Env = [
          "PYTHONPATH=/nix/store/*-instantmesh-0.1.0/lib/python3.10/site-packages"
          "LD_LIBRARY_PATH=/usr/lib64:/lib/x86_64-linux-gnu:${pkgs.lib.makeLibraryPath (commonDeps pkgs)}"
        ] ++ (if backend == "cuda" then [
          "NVIDIA_VISIBLE_DEVICES=all"
          "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
          "CUDA_PATH=/usr/local/cuda"
        ] else [
          "HIP_VISIBLE_DEVICES=0"
          "ROCM_PATH=/opt/rocm"
        ]);
      };
    };

  in {
    packages = {
      instantmesh-cuda = mkInstantMeshPackage { pkgs = pkgsCuda; backend = "cuda"; };
      instantmesh-rocm = mkInstantMeshPackage { pkgs = pkgsRocm; backend = "rocm"; };

      instantmesh-cuda-docker = mkInstantMeshDocker {
        pkgs = pkgsCuda;
        base = cudaBase;
        backend = "cuda";
        package = self.packages.${system}.instantmesh-cuda;
      };
      instantmesh-rocm-docker = mkInstantMeshDocker {
        pkgs = pkgsRocm;
        base = rocmBase;
        backend = "rocm";
        package = self.packages.${system}.instantmesh-rocm;
      };
    };

    apps = {
      instantmesh-cuda-lock = {
        type = "app";
        program = "${self.packages.${system}.instantmesh-cuda.lock}";
      };
      instantmesh-rocm-lock = {
        type = "app";
        program = "${self.packages.${system}.instantmesh-rocm.lock}";
      };
    };

    devShells = {
      cuda = pkgsCuda.mkShell rec {
        name = "instantmesh-cuda";
        nativeBuildInputs = commonDeps pkgsCuda ++ [ pkgsCuda.python310 pkgsCuda.python310Packages.venvShellHook ];
        buildInputs = with pkgsCuda; [ cudatoolkit linuxPackages.nvidia_x11 cudaPackages.cudnn ];
        venvDir = "./.venv-cuda";
        postVenv = ''
          pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
          pip install -U xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
          pip install -U -r InstantMesh/requirements.txt
        '';
        shellHook = ''
          if [ ! -d "InstantMesh" ]; then
            git clone https://github.com/TencentARC/InstantMesh.git
          fi
          export LD_LIBRARY_PATH=${pkgsCuda.lib.makeLibraryPath buildInputs}:${pkgsCuda.stdenv.cc.cc.lib}/lib
          export CUDA_PATH=${pkgsCuda.cudatoolkit}
          export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgsCuda.cudatoolkit}
        '';
      };

      rocm = pkgsRocm.mkShell rec {
        name = "instantmesh-rocm";
        nativeBuildInputs = commonDeps pkgsRocm ++ [ pkgsRocm.python310 pkgsRocm.python310Packages.venvShellHook ];
        buildInputs = with pkgsRocm; [ rocmPackages.rocm-runtime rocmPackages.miopen ];
        venvDir = "./.venv-rocm";
        postVenv = ''
          pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0/
          pip install -U xformers==0.0.22.post7 || pip install git+https://github.com/facebookresearch/xformers.git@0.0.22.post7
          pip install -U -r InstantMesh/requirements.txt
        '';
        shellHook = ''
          if [ ! -d "InstantMesh" ]; then
            git clone https://github.com/TencentARC/InstantMesh.git
          fi
          export LD_LIBRARY_PATH=${pkgsRocm.lib.makeLibraryPath buildInputs}:${pkgsRocm.stdenv.cc.cc.lib}/lib
        '';
      };
    };
  });
}
