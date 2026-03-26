#!/bin/bash
# CTranslate2 ROCm Build Script
# Automated build for AMD GPUs with ROCm support

set -e  # Exit on error

# Add ROCm to PATH
export PATH=/opt/rocm/bin:/opt/rocm-6.3.0/bin:$PATH
export ROCM_PATH=/opt/rocm

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CTranslate2 ROCm Build Script ===${NC}"

# Detect GPU architecture
detect_gpu_arch() {
    if command -v rocminfo &> /dev/null; then
        ARCH=$(rocminfo | grep gfx | head -1 | awk '{print $2}')
        echo -e "${GREEN}Detected GPU architecture: ${ARCH}${NC}"
    else
        echo -e "${YELLOW}Warning: rocminfo not found. Using default gfx1100${NC}"
        ARCH="gfx1100"
    fi
}

# Check ROCm installation
check_rocm() {
    echo -e "${YELLOW}Checking ROCm installation...${NC}"

    if ! command -v rocm-smi &> /dev/null; then
        echo -e "${RED}Error: ROCm not found. Please install ROCm first.${NC}"
        exit 1
    fi

    ROCM_VERSION=$(rocm-smi --showdriverversion | grep "Driver version" | awk '{print $3}')
    echo -e "${GREEN}ROCm driver version: ${ROCM_VERSION}${NC}"

    # Check required packages
    MISSING_PKGS=()
    for pkg in miopen-hip hipblas-dev rocblas-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg"; then
            MISSING_PKGS+=($pkg)
        fi
    done

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo -e "${RED}Missing required packages: ${MISSING_PKGS[*]}${NC}"
        echo -e "${YELLOW}Install with: sudo apt-get install ${MISSING_PKGS[*]}${NC}"
        exit 1
    fi

    echo -e "${GREEN}All ROCm dependencies found!${NC}"
}

# Configure with CMake
configure_cmake() {
    echo -e "${YELLOW}Configuring with CMake...${NC}"

    rm -rf build
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DWITH_DNNL=OFF \
        -DWITH_OPENBLAS=OFF \
        -DOPENMP_RUNTIME=NONE \
        -DENABLE_CPU_DISPATCH=OFF \
        -DCMAKE_PREFIX_PATH="/opt/rocm" \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DAMDGPU_TARGETS=${ARCH} \
        -DBUILD_CLI=OFF

    # Verify configuration
    if grep -q "WITH_CUDNN:BOOL=ON" CMakeCache.txt; then
        echo -e "${GREEN}✓ MIOpen (cuDNN) support enabled${NC}"
    else
        echo -e "${RED}✗ MIOpen support NOT enabled. Check CMake output above.${NC}"
        exit 1
    fi

    if grep -q "ENABLE_CPU_DISPATCH:BOOL=OFF" CMakeCache.txt; then
        echo -e "${GREEN}✓ CPU dispatch disabled (GPU-only build)${NC}"
    else
        echo -e "${RED}✗ CPU dispatch NOT disabled. GPU-only build failed.${NC}"
        exit 1
    fi

    cd ..
}

# Build library
build_library() {
    echo -e "${YELLOW}Building CTranslate2 library...${NC}"

    cd build
    make -j28

    # Verify MIOpen linkage
    if ldd libctranslate2.so | grep -q "libMIOpen"; then
        echo -e "${GREEN}✓ Library successfully linked with MIOpen${NC}"
    else
        echo -e "${RED}✗ MIOpen not linked. Build may be incomplete.${NC}"
        exit 1
    fi

    # Verify GPU-only build (no AVX CPU kernels)
    AVX_COUNT=$(nm libctranslate2.so | grep -i avx | wc -l)
    if [ $AVX_COUNT -eq 0 ]; then
        echo -e "${GREEN}✓ GPU-only build confirmed (no AVX symbols)${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Found ${AVX_COUNT} AVX symbols (CPU kernels present)${NC}"
    fi

    cd ..
}

# Install system-wide
install_library() {
    echo -e "${YELLOW}Installing library system-wide...${NC}"
    echo -e "${YELLOW}This requires sudo password${NC}"

    cd build
    sudo make install
    sudo ldconfig

    echo -e "${GREEN}✓ Library installed to /usr/local/lib${NC}"
    cd ..
}

# Build Python wheel
build_python() {
    echo -e "${YELLOW}Building Python wheel...${NC}"

    cd python
    rm -rf build dist

    # Set environment variable
    export CTRANSLATE2_ROOT=$(pwd)/../build

    # Check for pybind11
    if ! python3 -c "import pybind11" 2>/dev/null; then
        echo -e "${YELLOW}Installing pybind11...${NC}"
        pip3 install pybind11 setuptools wheel
    fi

    python3 setup.py bdist_wheel

    WHEEL=$(ls dist/*.whl)
    echo -e "${GREEN}✓ Wheel created: ${WHEEL}${NC}"

    cd ..
}

# Verify final package
verify_package() {
    echo -e "${YELLOW}Verifying Python package...${NC}"

    cd python
    TEMP_VENV="/tmp/ct2_test_venv"

    # Create temporary venv
    python3 -m venv $TEMP_VENV
    source $TEMP_VENV/bin/activate

    # Install wheel
    pip install dist/*.whl

    # Verify MIOpen linkage
    python -c "
import ctranslate2._ext
import subprocess
so_path = ctranslate2._ext.__file__
result = subprocess.run(['ldd', so_path], capture_output=True, text=True)
if 'libMIOpen.so' in result.stdout:
    print('✓ MIOpen successfully linked')
    exit(0)
else:
    print('✗ MIOpen NOT found in Python package')
    exit(1)
"

    STATUS=$?
    deactivate
    rm -rf $TEMP_VENV

    cd ..

    if [ $STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ Package verification successful!${NC}"
    else
        echo -e "${RED}✗ Package verification failed${NC}"
        exit 1
    fi
}

# Main execution
main() {
    detect_gpu_arch
    check_rocm
    configure_cmake
    build_library
    install_library
    build_python
    verify_package

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Install the wheel with:"
    echo -e "  ${YELLOW}pip install python/dist/*.whl${NC}"
    echo ""
    echo -e "Or upload to GitHub releases"
}

# Run main
main
