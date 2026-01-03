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
#include "chpl-comm.h"


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
typedef struct __nv_bfloat16_raw __nv_bfloat16;
typedef struct __half_raw        __half;
typedef struct __half_raw        half;
#endif

int chpl_gpu_impl_pgas_enabled(void) {
  static int enabled = -1;
  if (enabled == -1) {
    const char* env = chpl_env_rt_get("GPU_PGAS_ENABLE", "false");
    printf("chpl_gpu_impl_pgas_enabled: env=%s\n", env);
    if (strcmp(env, "true") == 0 || strcmp(env, "1") == 0) {
      enabled = 1;
    } else {
      enabled = 0;
    }
  }
  return enabled;
}

// Defined in ChapelGpuSupport.chpl
extern "C" void chpl_gpu_pgas_init_inner(void);

void chpl_gpu_impl_pgas_setup(void) {
  u_int8_t uid[128];
  u_int8_t my_uid[128];

  // Broadcast the unique ID to all nodes
  if (chpl_nodeID == 0) {
    printf("chpl_gpu_impl_pgas_setup: broadcasting unique ID from node 0\n");
    chpl_gpu_pgas_init_inner();
    printf("chpl_gpu_impl_pgas_setup: setup complete\n");
  }
}

void chpl_gpu_pgas_get_uid(u_int8_t uid_out[128]) {
  printf("chpl_gpu_pgas_get_uid: Getting NVSHMEM unique ID\n");
  nvshmemx_uniqueid_t* uid_ptr = (nvshmemx_uniqueid_t*)uid_out;
  *uid_ptr = NVSHMEMX_UNIQUEID_INITIALIZER;
  int status;
  status = nvshmemx_get_uniqueid(uid_ptr);
  if (status != 0) {
    fprintf(stderr,
            "[%s] Failed to get NVSHMEM unique ID, status: %d\n",
            __func__, status);
    exit(1);
  }
}

void chpl_gpu_pgas_init_with_uid(int rank, int nranks, u_int8_t uid_in[128]) {
  printf("chpl_gpu_pgas_init_with_uid: Initializing NVSHMEM on rank %d/%d\n",
         rank, nranks);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

  nvshmemx_set_attr_uniqueid_args(rank, nranks, (nvshmemx_uniqueid_t*)uid_in, &attr);

  printf("chpl_gpu_pgas_init_with_uid: Calling nvshmemx_hostlib_init_attr\n");

  int status;
  status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  // status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  printf("chpl_gpu_pgas_init_with_uid: NVSHMEM initialized on rank %d/%d\n",
         rank, nranks);
  if (status != 0) {
    fprintf(stderr,
            "[%s] Failed to initialize NVSHMEM with unique ID, status: %d\n",
            __func__, status);
    exit(1);
  }
}

void* chpl_gpu_impl_pgas_sym_malloc(const size_t size) {
  return nvshmem_malloc(size);
}

void chpl_gpu_impl_pgas_sym_free(void* ptr) {
  nvshmem_free(ptr);
}

void chpl_gpu_impl_pgas_comm_get(void *dst, c_nodeid_t node, void* src,
                    size_t size) {
  printf("chpl_gpu_impl_pgas_comm_get: node=%d size=%zu\n", (int)node, size);
  nvshmem_getmem(dst, src, size, (int)node);
  // nvshmem_quiet();
}

void chpl_gpu_impl_pgas_comm_put(void* dst, c_nodeid_t node, void* src,
                    size_t size) {
  printf("chpl_gpu_impl_pgas_comm_put: node=%d size=%zu\n", (int)node, size);
  nvshmem_putmem(dst, src, size, (int)node);
  // nvshmem_quiet();
}
