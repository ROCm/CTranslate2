#pragma once
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_bfloat16.h>
    #include <hipcub/hipcub.hpp>

    // ROCm bfloat16 math functions for device code only
    // These use hip_bfloat16's operator float() for conversions
    __device__ inline hip_bfloat16 hsin(const hip_bfloat16 h) {
        return hip_bfloat16(__sinf(float(h)));
    }

    __device__ inline hip_bfloat16 hcos(const hip_bfloat16 h) {
        return hip_bfloat16(__cosf(float(h)));
    }

    __device__ inline hip_bfloat16 hexp(const hip_bfloat16 h) {
        return hip_bfloat16(__expf(float(h)));
    }

    __device__ inline hip_bfloat16 hlog(const hip_bfloat16 h) {
        return hip_bfloat16(__logf(float(h)));
    }

    __device__ inline hip_bfloat16 __habs(const hip_bfloat16 a) {
        hip_bfloat16 ret = a;
        ret.data &= 0x7FFF;
        return ret;
    }

    __device__ inline hip_bfloat16 __hmax(const hip_bfloat16 a, const hip_bfloat16 b) {
        return hip_bfloat16(fmaxf(float(a), float(b)));
    }

    // hipcub FpLimits -> NumericTraits compatibility wrapper
    namespace hipcub {
        template <typename T>
        struct FpLimits {
            static __host__ __device__ __forceinline__ T Lowest() {
                return NumericTraits<T>::Lowest();
            }
            static __host__ __device__ __forceinline__ T Max() {
                return NumericTraits<T>::Max();
            }
        };
    }
#endif
