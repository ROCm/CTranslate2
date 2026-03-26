# Building CTranslate2 for AMD ROCm

This guide provides step-by-step instructions for compiling CTranslate2 with full support for AMD ROCm, enabling high-performance inference on AMD GPUs.

## Quick Build (Recommended)

For most users, the automated build script is the recommended method. It handles dependency checks, GPU detection, and wheel creation.

```bash
git clone https://github.com/dapovoa/CTranslate2.git
cd CTranslate2
./build_rocm.sh
```

For manual configuration and a deeper understanding of the build process, follow the detailed steps below.

---

## Manual Build Instructions

### 1. Prerequisites

#### System Requirements
*   **GPU**: An AMD GPU compatible with ROCm.
*   **OS**: Ubuntu 22.04/24.04 LTS or a compatible Linux distribution.
*   **ROCm**: Version 6.3.0 or later.
*   **Python**: Version 3.8 or later.

#### Install Dependencies
First, install the required ROCm packages.

```bash
# Update package lists
sudo apt-get update

# Install ROCm core libraries
sudo apt-get install -y \
    rocm-dev \
    rocm-libs \
    miopen-hip \
    miopen-hip-dev \
    hipblas-dev \
    rocblas-dev

# Verify the ROCm installation
rocm-smi --showdriverversion
```

### 2. Build the C++ Library

This involves cloning the repository and using CMake to configure and compile the project.

#### Step 2.1: Clone the Repository
```bash
git clone https://github.com/dapovoa/CTranslate2.git
cd CTranslate2
mkdir -p build && cd build
```

#### Step 2.2: Configure with CMake
The following command configures the build for a ROCm environment.

**Note:** You must specify your GPU architecture using the `-DCMAKE_HIP_ARCHITECTURES` flag. Find your architecture by running `rocminfo | grep gfx`.

```bash
cmake -DCMAKE_PREFIX_PATH="/opt/rocm" \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=OFF \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
      -DAMDGPU_TARGETS="gfx1100" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=HIP \
      -DENABLE_CPU_DISPATCH=OFF \
      -DCMAKE_CXX_FLAGS="-O3" ..
```

**Key CMake Flags:**
*   `-DWITH_CUDA=ON`: Enables GPU support (required for ROCm/HIP).
*   `-DWITH_CUDNN=ON`: Enables MIOpen support, which is crucial for models like Whisper (Conv1D).
*   `-DCMAKE_HIP_ARCHITECTURES="gfx1100"`: **(Important!)** Set this to your GPU's architecture (e.g., `gfx1030`, `gfx90a`).
*   `-DENABLE_CPU_DISPATCH=OFF`: Creates a GPU-only build for maximum performance.

#### Step 2.3: Compile and Install
```bash
# Compile the library
make -j$(nproc)

# Install the library system-wide
sudo make install
sudo ldconfig
```
This makes the `libctranslate2.so` library available to other applications, including the Python wrapper.

### 3. Build the Python Wheel

After installing the C++ library, you can build the Python package.

#### Step 3.1: Install Python Dependencies
```bash
cd ../python
pip install pybind11 setuptools wheel
```

#### Step 3.2: Build the Wheel
The `setup.py` in this fork is modified to automatically find and link the required ROCm libraries.

```bash
python setup.py bdist_wheel
```

#### Step 3.3: Install the Wheel
```bash
pip install dist/ctranslate2-*.whl
```

### 4. Verification

To ensure everything is working correctly, you can run a few quick checks.

#### Check Library Links
This script verifies that the Python extension is correctly linked against `libMIOpen`.

```python
import ctranslate2._ext
import subprocess

so_path = ctranslate2._ext.__file__
linked_libs = subprocess.check_output(['ldd', so_path]).decode()

if 'libMIOpen.so' in linked_libs:
    print("Successfully linked against MIOpen.")
else:
    print("MIOpen library not found. Please check the build process.")
```

#### Test a Whisper Model
This snippet demonstrates how to load a Whisper model and check for supported compute types. (Note: You need a CTranslate2-converted Whisper model for this to run).

```python
import ctranslate2
import numpy as np

if not ctranslate2.get_supported_compute_types('cuda'):
    raise RuntimeError("No GPU compute types available. The build may have failed.")

print("GPU support is enabled.")
print(f"Supported compute types: {ctranslate2.get_supported_compute_types('cuda')}")

# Example:
# generator = ctranslate2.Generator("whisper-large-v3-ct2", device="cuda")
# ...
```

### 5. Troubleshooting

**Q: I get the error "Conv1D on GPU currently requires the cuDNN library". What's wrong?**
**A:** This happens when MIOpen is not correctly linked.
1.  Ensure MIOpen is installed (`dpkg -l | grep miopen`).
2.  Verify you used `-DWITH_CUDNN=ON` during the CMake configuration.
3.  Delete the `build` directory and rebuild from scratch.

**Q: CMake fails with "hipblas not found".**
**A:** You are missing the hipBLAS development package. Install it with:
`sudo apt-get install hipblas-dev`

**Q: The Python wheel doesn't work or doesn't find the ROCm libraries.**
**A:**
1.  Confirm that you ran `sudo make install` and `sudo ldconfig` after compiling the C++ library.
2.  Check that `ldd $(which python) | grep libctranslate2` shows the correct path.
3.  Try rebuilding the wheel in a clean directory (`rm -rf build dist`).

### 6. ROCm Build Notes

#### Key Features
*   **Up-to-date:** Full compatibility with CTranslate2 v4.6.0.
*   **Accelerated Operations:** Includes support for Flash Attention v2 and other GPU-accelerated features.
*   **Wide Model Support:** Works with Whisper (large-v3-turbo), wav2vec2bert, and more.

#### Known Limitations
*   **AWQ (4-bit) Quantization**: This is **disabled** as it relies on NVIDIA-specific code (PTX). Using an AWQ model will result in a runtime error.
    *   **Alternative**: Use `int8` or `float16` quantization, which are fully supported.

#### Performance Recommendations
*   **Compute Type**: Use `int8_float16` or `float16` for the best balance of speed and quality.
*   **VRAM**: At least 12GB of VRAM is recommended for larger models like Whisper-large-v3.
*   **ROCm Version**: Use ROCm 6.3.0 or newer for the best performance and stability.

---
*This guide is maintained by [@dapovoa](https://github.com/dapovoa) as part of the CTranslate2 ROCm port.*