#pragma once
// Type mappings for CUDA to HIP compatibility (safe for C++ compilation)
#ifdef __HIP_PLATFORM_AMD__
  // cuBLAS types
  #define cublasStatus_t hipblasStatus_t
  #define cublasHandle_t hipblasHandle_t

  // cuBLAS status codes
  #define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
  #define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
  #define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
  #define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
  #define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
  #define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
  #define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
  #define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
  #define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
  #define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_UNKNOWN

  // CUDA runtime types
  #define cudaStream_t hipStream_t
  #define cudaError_t hipError_t
  #define cudaEvent_t hipEvent_t
  #define cudaDeviceProp hipDeviceProp_t

  // CUDA runtime constants
  #define cudaSuccess hipSuccess
  #define cudaStreamDefault hipStreamDefault
  #define cudaGetErrorString hipGetErrorString
  #define cudaGetDevice hipGetDevice
  #define cudaStreamCreate hipStreamCreate
  #define cudaStreamDestroy hipStreamDestroy

  // cuDNN types (MIOpen equivalents)
  #ifdef CT2_WITH_CUDNN
    #define cudnnHandle_t miopenHandle_t
    #define cudnnDataType_t miopenDataType_t
    #define cudnnStatus_t miopenStatus_t
    #define cudnnTensorDescriptor_t miopenTensorDescriptor_t
    #define cudnnFilterDescriptor_t miopenTensorDescriptor_t
    #define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t

    // cuDNN status codes
    #define CUDNN_STATUS_SUCCESS miopenStatusSuccess

    // cuDNN functions
    #define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
    #define cudnnSetTensor4dDescriptor miopenSet4dTensorDescriptor
    #define cudnnCreateFilterDescriptor miopenCreateTensorDescriptor
    #define cudnnSetFilter4dDescriptor miopenSet4dTensorDescriptor
    #define cudnnGetErrorString miopenGetErrorString
  #endif
#endif
