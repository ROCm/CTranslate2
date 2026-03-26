#pragma once

#ifdef __HIP_PLATFORM_AMD__
  #include <hiprand/hiprand_kernel.h>
  #define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#else
  #include <curand_kernel.h>
#endif

namespace ctranslate2 {
  namespace cuda {

    curandStatePhilox4_32_10_t* get_curand_states(size_t num_states);

  }
}
