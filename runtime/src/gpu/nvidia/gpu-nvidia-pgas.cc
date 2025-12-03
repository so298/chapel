#include "sys_basic.h"
#include "chplrt.h"
#include "chpl-mem.h"
#include "chpl-gpu.h"
#include "chpl-gpu-impl.h"
#include "chpl-linefile-support.h"
#include "chpl-tasks.h"
#include "error.h"
#include "chplcgfns.h"
#include "chpl-env-gen.h"
#include "chpl-env.h"
#include "chpl-topo.h"


#include "chpl-gpu.h"
#include "chpl-gpu-impl.h"

#include <cuda.h>
#include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>

// #include "../common/cuda-utils.h"

#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
// #include <nvshmem.h>
// #include <nvshmemx.h>

#ifndef __cplusplus
// C のときだけ、NVSHMEM が期待する名前に typedef してやる
typedef struct __nv_bfloat16_raw __nv_bfloat16;
typedef struct __half_raw        __half;
typedef struct __half_raw        half;
#endif


void chpl_gpu_impl_setup_pgas(void) {
  // nvshmem_init();
  nvshmemx_hostlib_init_attr(0, NULL);
}

