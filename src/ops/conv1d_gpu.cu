#include "ctranslate2/ops/conv1d.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output) const {
#ifndef CT2_WITH_CUDNN
      (void)input;
      (void)weight;
      (void)bias;
      (void)output;
      throw std::runtime_error("Conv1D on GPU currently requires the cuDNN library "
                               "which is not integrated in this build");

#else
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      cudnnDataType_t data_type = cuda::get_cudnn_data_type(input.dtype());

      cudnnTensorDescriptor_t input_desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, 
#ifndef __HIP_PLATFORM_AMD__			      
			      CUDNN_TENSOR_NCHW, 
#endif			      
			      data_type,  batch_size, in_channels, 1, input_length));

      cudnnTensorDescriptor_t output_desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, 
#ifndef __HIP_PLATFORM_AMD__
			      CUDNN_TENSOR_NCHW, 
#endif			      
			      data_type,  batch_size, out_channels, 1, output_length));

      cudnnFilterDescriptor_t weight_desc;
      CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(weight_desc, data_type, 
#ifndef __HIP_PLATFORM_AMD__			      
			      CUDNN_TENSOR_NCHW,
#endif			      
      	                               out_channels, in_channels, 1, kernel_size));

      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
#ifndef __HIP_PLATFORM_AMD__      
      CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                  /*pad_h=*/0, /*pad_w=*/_padding,
                                                  /*stride_h=*/1, /*stride_w=*/_stride,
                                                  /*dilation_h=*/1, /*dilation_w=*/_dilation,
                                                  CUDNN_CROSS_CORRELATION,
                                                  CUDNN_DATA_FLOAT));
#else
      CUDNN_CHECK(miopenInitConvolutionDescriptor(conv_desc,
                                                  miopenConvolution,
                                                  /*pad_h=*/0, /*pad_w=*/_padding,
                                                  /*stride_h=*/1, /*stride_w=*/_stride,
                                                  /*dilation_h=*/1, /*dilation_w=*/_dilation) );
#endif      
#ifndef __HIP_PLATFORM_AMD__
      CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));
      if (data_type == CUDNN_DATA_HALF)
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
      cudnnHandle_t handle = cuda::get_cudnn_handle();

#ifndef __HIP_PLATFORM_AMD__
      cudnnConvolutionFwdAlgo_t algo = (bias
                                        ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                                        : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
#else
      miopenConvFwdAlgorithm_t algo = (bias
                                        ? miopenConvolutionFwdAlgoGEMM
                                        : miopenConvolutionFwdAlgoImplicitGEMM);
#endif


      size_t workspace_size = 0;
      void* workspace = nullptr;
#ifndef __HIP_PLATFORM_AMD__

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                          input_desc,
                                                          weight_desc,
                                                          conv_desc,
                                                          output_desc,
                                                          algo,
                                                          &workspace_size));

      if (workspace_size > 0)
        workspace = get_allocator<Device::CUDA>().allocate(workspace_size);
#else
              std::size_t count;
        CUDNN_CHECK(miopenConvolutionForwardGetSolutionCount(handle,
                                                 weight_desc,
                                                 input_desc,
                                                 conv_desc,
                                                 output_desc,
                                                 &count));
        if(count <1){
                std::cout<<"count: "<<count<<std::endl;
                return;
        }
        auto solutions = std::vector<miopenConvSolution_t>(count);
        CUDNN_CHECK(miopenConvolutionForwardGetSolution(handle,
                                             weight_desc,
                                             input_desc,
                                             conv_desc,
                                             output_desc,
                                             count,
                                             &count,
                                             solutions.data()));
        const miopenConvSolution_t* selected = &solutions.front();
        CUDNN_CHECK(miopenConvolutionForwardGetSolutionWorkspaceSize(handle,
                                                                     weight_desc,
                                                                     input_desc,
                                                                     conv_desc,
                                                                     output_desc,
                                                                     selected->solution_id,
                                                                     &workspace_size));
        if (workspace_size > 0){
                workspace = get_allocator<Device::CUDA>().allocate(workspace_size);
        }
        CUDNN_CHECK(miopenConvolutionForwardCompileSolution(handle,
                                                 weight_desc,
                                                 input_desc,
                                                 conv_desc,
                                                 output_desc,
                                                 selected->solution_id));

        CUDNN_CHECK(miopenConvolutionForwardImmediate(handle,
                                                     weight_desc,
                                                     weight.buffer(),
                                                     input_desc,
                                                     input.buffer(),
                                                     conv_desc,
                                                     output_desc,
                                                     output.buffer(),
                                                     workspace,
                                                     workspace_size,
                                                     selected->solution_id));
#endif 	
      float alpha = 1;
      float beta = 0;

      if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, 
#ifndef __HIP_PLATFORM_AMD__
				CUDNN_TENSOR_NCHW, 
#endif				
				data_type, 1, out_channels, 1, 1));


        cudnnActivationDescriptor_t activation_desc;
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
#ifndef __HIP_PLATFORM_AMD__	
        CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc,
                                                 CUDNN_ACTIVATION_IDENTITY,
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 /*coef=*/0));
#else
        CUDNN_CHECK(miopenSetActivationDescriptor(activation_desc,
                                                 miopenActivationPASTHRU,
                                                 0,
                                                 0,
                                                 0));
#endif	
#ifndef __HIP_PLATFORM_AMD__	
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(handle,
                                                          &alpha,
                                                          input_desc,
                                                          input.buffer(),
                                                          weight_desc,
                                                          weight.buffer(),
                                                          conv_desc,
                                                          algo,
                                                          workspace,
                                                          workspace_size,
                                                          &beta,
                                                          output_desc,
                                                          output.buffer(),
                                                          bias_desc,
                                                          bias->buffer(),
                                                          activation_desc,
                                                          output_desc,
                                                          output.buffer()));
#else
        CUDNN_CHECK(miopenConvolutionForwardBias(handle,
                                     &alpha,
                                     bias_desc,
                                     bias->buffer(),
                                     &beta,
                                     output_desc,
                                     output.buffer()));
#endif	

        CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));

      } else {
#ifndef __HIP_PLATFORM_AMD__	      
        CUDNN_CHECK(cudnnConvolutionForward(handle,
                                            &alpha,
                                            input_desc,
                                            input.buffer(),
                                            weight_desc,
                                            weight.buffer(),
                                            conv_desc,
                                            algo,
                                            workspace,
                                            workspace_size,
                                            &beta,
                                            output_desc,
                                            output.buffer()));
#endif	
      }

      if (workspace)
        get_allocator<Device::CUDA>().free(workspace);

      CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
#endif
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Conv1D::compute<Device::CUDA, T>(const StorageView& input,          \
                                     const StorageView& weight,         \
                                     const StorageView* bias,           \
                                     StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
