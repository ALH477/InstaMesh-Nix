# InstantMesh Nix Flake

Efficient, reproducible Nix environments and Docker images for running [InstantMesh](https://github.com/TencentARC/InstantMesh): a state-of-the-art feed-forward framework for high-quality 3D mesh generation from a single input image.

This flake provides:
- Development shells for NVIDIA (CUDA) and AMD (ROCm) GPUs
- Pure Nix-packaged InstantMesh installation (via dream2nix for Python dependencies)
- Buildable multi-layer Docker images with GPU support (using `dockerTools`)

Copyright © 2026 DeMoD (ALH477)  
All rights reserved.

## Features

- Supports **NVIDIA CUDA** (12.1+) and **AMD ROCm** (6.0+)
- Virtual environments created on first entry (venv + pip installs)
- Clones the official InstantMesh repo automatically
- Docker images based on official NVIDIA/ROCm base images
- CLI inference, Gradio web UI, textured mesh export, video preview
- Reproducible via lock files (generated with dream2nix)

## Prerequisites

- Nix with flakes enabled (`experimental-features = nix-command flakes`)
- For **CUDA**: NVIDIA drivers + CUDA toolkit installed on host
- For **ROCm**: AMD ROCm drivers installed on host
- For **Docker**: Docker daemon running; NVIDIA Container Toolkit (for CUDA) or ROCm Docker support (for ROCm)

On NixOS, enable the appropriate hardware modules in your configuration:
```nix
# For NVIDIA
hardware.nvidia.enable = true;
services.xserver.videoDrivers = [ "nvidia" ];

# For ROCm (AMD)
hardware.amdgpu.amdvlk.enable = true;
```

## Quick Start (Development Shell)

```bash
# Enter CUDA shell (creates .venv-cuda + installs deps on first run)
nix develop .#cuda

# Or ROCm shell
nix develop .#rocm
```

Inside the shell:
```bash
cd InstantMesh

# Run Gradio demo (open http://127.0.0.1:7860 in browser)
python app.py

# Or CLI inference (upload your own image)
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --export_texmap
```

## Building & Using Docker Images

First build the image (may take time the first run due to layers):

```bash
# NVIDIA / CUDA
nix build .#instantmesh-cuda-docker

# AMD / ROCm
nix build .#instantmesh-rocm-docker
```

Load into Docker:
```bash
docker load < result
```

Run (mount current directory as workspace):

```bash
# NVIDIA
docker run --rm -it --gpus all -v $(pwd):/workspace instantmesh-cuda:latest

# ROCm (adjust devices as needed)
docker run --rm -it --device /dev/kfd --device /dev/dri -v $(pwd):/workspace instantmesh-rocm:latest
```

Inside container:
```bash
instantmesh-app                # Start Gradio UI
instantmesh-run configs/instant-mesh-large.yaml /workspace/your-image.png --output /workspace/output/
```

## License

BSD 3-Clause License

Copyright © 2026 DeMoD (ALH477)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Original InstantMesh License & Citation

The underlying InstantMesh project (TencentARC) is licensed under **Apache-2.0**.

Please cite the original work if you use this in research:

```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

## Troubleshooting

- Hash mismatches on base images/repo fetch? Replace placeholders in `flake.nix` with hashes from the Nix error message and rebuild.
- xformers fails on ROCm? The flake attempts source build fallback.
- GPU not detected in container? Verify host container toolkit setup.
- Model download slow? First inference auto-downloads from Hugging Face.

Contributions, issues, and pull requests welcome!

Maintained by DeMoD (@DeMoDLLC / ALH477)
