#pragma once
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include<hip/hip_bfloat16.h>
    #define __nv_bfloat16 hip_bfloat16
    #define  CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
    #define  CUBLAS_GEMM_DEFAULT   HIPBLAS_GEMM_DEFAULT
     __device__ hip_bfloat16 hlog(const hip_bfloat16 h) {
 	 return hip_bfloat16(__ocml_log_f32(float(h)));
     }
     __device__ hip_bfloat16 hsin(const hip_bfloat16 h) {
  	return hip_bfloat16(__ocml_sin_f32(float(h)));
    }
    __device__ hip_bfloat16 hcos(const hip_bfloat16 h) {
  	return hip_bfloat16(__ocml_cos_f32(float(h)));
    }
    __device__ hip_bfloat16 hexp(const hip_bfloat16 h) {
      return hip_bfloat16(__ocml_exp_f32(float(h)));
    }

    __device__ hip_bfloat16 __habs(const hip_bfloat16 a) {
  	auto ret = a;
        ret.data &= 0x7FFF;
  	return ret;
    }
    __device__ hip_bfloat16 __hmax(const hip_bfloat16 a, const hip_bfloat16 b) {
     	 return hip_bfloat16(__ocml_fmax_f32(float(a), float(b)));
    }
  #define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
  #define cublasStatus_t  hipblasStatus_t 
  #define cublasHandle_t  hipblasHandle_t 
  #define curand_init  hiprand_init
    #define cudaDeviceProp hipDeviceProp_t
    #define cudaDeviceSynchronize hipDeviceSynchronize
    #define cudaErrorInsufficientDriver hipErrorInsufficientDriver
    #define cudaErrorNoDevice hipErrorNoDevice
    #define cudaError_t hipError_t
    #define cudaEventCreate hipEventCreate
    #define cudaEventElapsedTime hipEventElapsedTime
    #define cudaEventRecord hipEventRecord
    #define cudaEventSynchronize hipEventSynchronize
    #define cudaEvent_t hipEvent_t
    #define cudaFree hipFree
    #define cudaGetDevice hipGetDevice
    #define cudaGetDeviceCount hipGetDeviceCount
    #define cudaGetDeviceProperties hipGetDeviceProperties
    #define cudaGetErrorString hipGetErrorString
    #define cudaGetLastError hipGetLastError
    #define cudaLaunchKernelGGL hipLaunchKernelGGL
    #define cudaMalloc hipMalloc
    #define cudaMemcpy hipMemcpy
    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
    #define cudaSuccess hipSuccess
    #define cudaMallocHost hipHostMalloc
    #define cudaStream_t hipStream_t
    #define cudaStreamCreate hipStreamCreate
    #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
    #define cudaStreamNonBlocking hipStreamNonBlocking
    #define cudaStreamDestroy hipStreamDestroy
    #define cudaSetDevice hipSetDevice
    #define udaMemcpyToSymbol hipMemcpyToSymbol c
    #define cudaMemcpyAsync  hipMemcpyAsync
    #define cudaFreeHost hipHostFree
    #define cudaDeviceReset hipDeviceReset
    #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define cudaStreamSynchronize  hipStreamSynchronize
    #define cudaMallocAsync hipMallocAsync
    #define cudaFreeAsync hipFreeAsync
    #define cudaDeviceGetAttribute hipDeviceGetAttribute
    #define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported 
    #define CUBLAS_OP_T HIPBLAS_OP_T 
    #define CUBLAS_OP_N HIPBLAS_OP_N
    #define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
    #define cublasSgemmStridedBatched hipblasSgemmStridedBatched
    #define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
    #define cublasSgemm  hipblasSgemm
    #define cublasGemmEx hipblasGemmEx
    #define CUDA_R_16F HIPBLAS_R_16F    
    #define CUDA_R_32F HIPBLAS_R_32F
    #define CUDA_R_16B HIPBLAS_R_16B
    #define CUDA_R_16BF HIPBLAS_R_16B
    #define CUDA_R_32I HIPBLAS_R_32I
    #define CUDA_R_8I HIPBLAS_R_8I
    #define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
    #define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
    #define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
    #define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
    #define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
    #define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
    #define CUBLAS_STATUS_INTERNAL_ERROR   HIPBLAS_STATUS_INTERNAL_ERROR
    #define CUBLAS_STATUS_NOT_SUPPORTED    HIPBLAS_STATUS_NOT_SUPPORTED
    #define CUBLAS_STATUS_LICENSE_ERROR    HIPBLAS_STATUS_UNKNOWN
    #define cublasCreate hipblasCreate
    #define cublasSetStream hipblasSetStream
    #define cublasDestroy hipblasDestroy
    #define cudaDataType_t hipblasDatatype_t
    #define cub hipcub
    #define cudaStreamDefault hipStreamDefault
    #define curand_uniform hiprand_uniform
//cudnn vs miopen
  #define cudnnHandle_t miopenHandle_t
  #define cudnnDataType_t miopenDataType_t
  #define cudnnStatus_t miopenStatus_t
  #define cudnnGetErrorString miopenGetErrorString
  #define cudnnCreate miopenCreate
  #define cudnnSetStream miopenSetStream
  #define cudnnDestroy miopenDestroy
  #define CUDNN_STATUS_SUCCESS miopenStatusSuccess
  #define CUDNN_DATA_FLOAT miopenFloat
  #define CUDNN_DATA_HALF miopenHalf
  #define CUDNN_DATA_BFLOAT16 miopenBFloat16
  #define CUDNN_DATA_BFLOAT16 miopenBFloat16
  #define CUDNN_DATA_INT32 miopenInt32
  #define CUDNN_DATA_INT8 miopenInt8
  #define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
  #define cudnnTensorDescriptor_t miopenTensorDescriptor_t
  #define cudnnSetTensor4dDescriptor miopenSet4dTensorDescriptor
  #define cudnnFilterDescriptor_t  miopenTensorDescriptor_t
  #define cudnnCreateFilterDescriptor  miopenCreateTensorDescriptor
  #define cudnnSetFilter4dDescriptor miopenSet4dTensorDescriptor
  #define cudnnActivationDescriptor_t miopenActivationDescriptor_t
  #define cudnnCreateActivationDescriptor miopenCreateActivationDescriptor
  #define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
  #define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
  #define cudnnDestroyActivationDescriptor miopenDestroyActivationDescriptor
  #define cudnnDestroyTensorDescriptor miopenDestroyTensorDescriptor
  #define cudnnDestroyConvolutionDescriptor miopenDestroyConvolutionDescriptor
  #define cudnnConvolutionBiasActivationForward miopenConvolutionBiasActivationForward
  #define cudnnDestroyFilterDescriptor miopenDestroyTensorDescriptor
#endif

