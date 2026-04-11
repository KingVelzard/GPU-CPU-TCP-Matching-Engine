#pragma once

#ifdef __CUDACC__
  #define DEVICE_ATTR __device__
#else
  #define DEVICE_ATTR
#endif
