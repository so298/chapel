/*
 * Copyright 2020-2025 Hewlett Packard Enterprise Development LP
 * Copyright 2004-2019 Cray Inc.
 * Other additional copyright holders may be indicated within.  *
 * The entirety of this work is licensed under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef HAS_GPU_LOCALE

// #define CHPL_GPU_ENABLE_PROFILE // define this before including chpl-gpu.h

#include "sys_basic.h"
#include "chplrt.h"
#include "chpl-mem.h"
#include "chpl-gpu.h"
#include "chpl-gpu-impl.h"
#include "chpl-linefile-support.h"
#include "chpl-tasks.h"
#include "error.h"
#include "chplcgfns.h"
#include "../common/cuda-utils.h"
#include "../common/cuda-shared.h"
#include "chpl-env-gen.h"
#include "chpl-env.h"
#include "chpl-topo.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
// GPU Arena Allocator
// ============================================================================
//
// This is a simple arena-based memory allocator for GPU device memory.
// It pre-allocates a large chunk of GPU memory at initialization time
// and serves allocation requests from this pool.
//
// Design goals:
// - Reduce overhead of frequent cuMemAlloc/cuMemFree calls
// - Allow future extension to more sophisticated algorithms
// - Each GPU device has its own independent arena
//
// The allocator uses a free list approach with block coalescing on free.
// Future implementations could replace this with buddy system, slab allocator,
// or other algorithms by implementing the chpl_gpu_arena_ops_t interface.
//
// To enable this allocator, set CHPL_RT_GPU_ARENA_ALLOCATOR=true at runtime.
// To disable it, set CHPL_RT_GPU_ARENA_ALLOCATOR=false.
// The heap size can be configured via CHPL_RT_GPU_HEAP_SIZE (default 1GiB).
// ============================================================================

// Default heap size: 1 GiB
#define CHPL_GPU_DEFAULT_HEAP_SIZE ((size_t)1 << 30)

// Minimum allocation alignment (16 bytes for GPU efficiency)
#define CHPL_GPU_ARENA_ALIGNMENT 16

// Minimum block size
#define CHPL_GPU_ARENA_MIN_BLOCK_SIZE 64

// ============================================================================
// Arena Block Metadata (Host-side)
// ============================================================================
// Block metadata is stored in host memory, not device memory.
// Device memory allocated via cuMemAlloc cannot be accessed from the host,
// so we maintain all bookkeeping structures on the host side.
//
// Each block tracks a region of device memory by offset and size.
// Blocks are linked in two ways:
//   1. By offset order (next/prev) - for coalescing adjacent free blocks
//   2. By free list (next_free/prev_free) - for fast allocation

typedef struct arena_block_s {
  size_t offset;                    // Offset from arena base in device memory
  size_t size;                      // Size of this block
  bool allocated;                   // Whether this block is allocated
  struct arena_block_s* next;       // Next block by offset order
  struct arena_block_s* prev;       // Previous block by offset order
  struct arena_block_s* next_free;  // Next free block (only valid if free)
  struct arena_block_s* prev_free;  // Previous free block (only valid if free)
} arena_block_t;

// ============================================================================
// Arena Allocator Operations Interface (for extensibility)
// ============================================================================
// This interface allows different allocator implementations to be plugged in.
// Current implementation: simple free list with coalescing
// Future possibilities: buddy system, slab allocator, etc.

struct chpl_gpu_arena_s;

typedef struct chpl_gpu_arena_ops_s {
  void* (*alloc)(struct chpl_gpu_arena_s* arena, size_t size);
  void (*free)(struct chpl_gpu_arena_s* arena, void* ptr);
  size_t (*get_alloc_size)(struct chpl_gpu_arena_s* arena, void* ptr);
  void (*destroy)(struct chpl_gpu_arena_s* arena);
} chpl_gpu_arena_ops_t;

// ============================================================================
// Arena Structure
// ============================================================================

typedef struct chpl_gpu_arena_s {
  void* base;                        // Base address of the arena memory (device pointer)
  size_t total_size;                 // Total size of the arena
  size_t used_size;                  // Currently used size (for statistics)
  arena_block_t* block_list;         // Head of all blocks (sorted by offset)
  arena_block_t* free_list;          // Head of the free list
  const chpl_gpu_arena_ops_t* ops;   // Allocator operations
  int device_id;                     // GPU device ID for this arena
  bool initialized;                  // Whether the arena is initialized
} chpl_gpu_arena_t;

// ============================================================================
// Free List Allocator Implementation (Host-side metadata)
// ============================================================================

static inline size_t align_up(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Convert device memory offset to pointer
static inline void* offset_to_ptr(chpl_gpu_arena_t* arena, size_t offset) {
  return (void*)((char*)arena->base + offset);
}

// Convert device memory pointer to offset
static inline size_t ptr_to_offset(chpl_gpu_arena_t* arena, void* ptr) {
  return (size_t)((char*)ptr - (char*)arena->base);
}

// Allocate a new block metadata structure (host memory)
static arena_block_t* block_create(size_t offset, size_t size) {
  arena_block_t* block = (arena_block_t*)chpl_mem_alloc(sizeof(arena_block_t),
                                                        CHPL_RT_MD_GPU_KERNEL_ARG, 0, 0);
  block->offset = offset;
  block->size = size;
  block->allocated = false;
  block->next = NULL;
  block->prev = NULL;
  block->next_free = NULL;
  block->prev_free = NULL;
  return block;
}

// Free a block metadata structure
static void block_destroy(arena_block_t* block) {
  chpl_mem_free(block, 0, 0);
}

// Remove block from free list
static void freelist_remove(chpl_gpu_arena_t* arena, arena_block_t* block) {
  if (block->prev_free) {
    block->prev_free->next_free = block->next_free;
  } else {
    arena->free_list = block->next_free;
  }
  if (block->next_free) {
    block->next_free->prev_free = block->prev_free;
  }
  block->next_free = NULL;
  block->prev_free = NULL;
}

// Insert block at head of free list
static void freelist_insert(chpl_gpu_arena_t* arena, arena_block_t* block) {
  block->next_free = arena->free_list;
  block->prev_free = NULL;
  if (arena->free_list) {
    arena->free_list->prev_free = block;
  }
  arena->free_list = block;
}

// Find block by device pointer
static arena_block_t* find_block_by_ptr(chpl_gpu_arena_t* arena, void* ptr) {
  size_t target_offset = ptr_to_offset(arena, ptr);
  arena_block_t* block = arena->block_list;
  while (block != NULL) {
    if (block->offset == target_offset) {
      return block;
    }
    block = block->next;
  }
  return NULL;
}

static void* freelist_alloc(chpl_gpu_arena_t* arena, size_t size) {
  // Align size and ensure minimum
  size_t aligned_size = align_up(size, CHPL_GPU_ARENA_ALIGNMENT);
  if (aligned_size < CHPL_GPU_ARENA_MIN_BLOCK_SIZE) {
    aligned_size = CHPL_GPU_ARENA_MIN_BLOCK_SIZE;
  }

  // First-fit search through free list
  arena_block_t* block = arena->free_list;
  while (block != NULL) {
    assert(!block->allocated);

    if (block->size >= aligned_size) {
      // Found a suitable block
      size_t remaining = block->size - aligned_size;

      // Check if we can split this block
      if (remaining >= CHPL_GPU_ARENA_MIN_BLOCK_SIZE) {
        // Split: create a new block for the remaining space
        arena_block_t* new_block = block_create(block->offset + aligned_size, remaining);

        // Insert new_block after block in offset-ordered list
        new_block->next = block->next;
        new_block->prev = block;
        if (block->next) {
          block->next->prev = new_block;
        }
        block->next = new_block;

        // Update original block's size
        block->size = aligned_size;

        // Replace block with new_block in free list
        new_block->next_free = block->next_free;
        new_block->prev_free = block->prev_free;
        if (new_block->prev_free) {
          new_block->prev_free->next_free = new_block;
        } else {
          arena->free_list = new_block;
        }
        if (new_block->next_free) {
          new_block->next_free->prev_free = new_block;
        }
        block->next_free = NULL;
        block->prev_free = NULL;
      } else {
        // Use the whole block, remove from free list
        freelist_remove(arena, block);
      }

      // Mark as allocated
      block->allocated = true;
      arena->used_size += block->size;

      return offset_to_ptr(arena, block->offset);
    }

    block = block->next_free;
  }

  // No suitable block found
  return NULL;
}

static void freelist_free(chpl_gpu_arena_t* arena, void* ptr) {
  if (ptr == NULL) return;

  // Find the block for this pointer
  arena_block_t* block = find_block_by_ptr(arena, ptr);
  if (block == NULL) {
    chpl_internal_error("GPU arena: invalid free - pointer not found in arena");
  }
  if (!block->allocated) {
    chpl_internal_error("GPU arena: double free detected");
  }

  // Mark as free
  block->allocated = false;
  arena->used_size -= block->size;

  // Try to coalesce with next block
  if (block->next != NULL && !block->next->allocated) {
    arena_block_t* next = block->next;

    // Remove next from free list
    freelist_remove(arena, next);

    // Merge next into block
    block->size += next->size;
    block->next = next->next;
    if (next->next) {
      next->next->prev = block;
    }

    // Free the merged block's metadata
    block_destroy(next);
  }

  // Try to coalesce with previous block
  if (block->prev != NULL && !block->prev->allocated) {
    arena_block_t* prev = block->prev;

    // Remove prev from free list
    freelist_remove(arena, prev);

    // Merge block into prev
    prev->size += block->size;
    prev->next = block->next;
    if (block->next) {
      block->next->prev = prev;
    }

    // Free block's metadata and use prev instead
    block_destroy(block);
    block = prev;
  }

  // Add to free list
  freelist_insert(arena, block);
}

static size_t freelist_get_alloc_size(chpl_gpu_arena_t* arena, void* ptr) {
  if (ptr == NULL) return 0;

  arena_block_t* block = find_block_by_ptr(arena, ptr);
  if (block == NULL || !block->allocated) {
    return 0;
  }
  return block->size;
}

static void freelist_destroy(chpl_gpu_arena_t* arena) {
  // Free all block metadata
  arena_block_t* block = arena->block_list;
  while (block != NULL) {
    arena_block_t* next = block->next;
    block_destroy(block);
    block = next;
  }
  arena->block_list = NULL;
  arena->free_list = NULL;
  arena->initialized = false;
}

static const chpl_gpu_arena_ops_t freelist_ops = {
  .alloc = freelist_alloc,
  .free = freelist_free,
  .get_alloc_size = freelist_get_alloc_size,
  .destroy = freelist_destroy
};

// ============================================================================
// Arena Initialization
// ============================================================================

static void arena_init(chpl_gpu_arena_t* arena, void* base, size_t size, int device_id) {
#if 0
  printf("Initializing GPU arena on device %d: base=%p, size=%zu bytes\n",
         device_id, base, size);
#endif

  arena->base = base;
  arena->total_size = size;
  arena->used_size = 0;
  arena->device_id = device_id;
  arena->ops = &freelist_ops;
  arena->initialized = true;

  // Initialize the entire arena as one big free block
  // Note: Block metadata is allocated in host memory, not device memory
  arena_block_t* initial_block = block_create(0, size);
  arena->block_list = initial_block;
  arena->free_list = initial_block;
}

// ============================================================================
// Per-device Arena Storage
// ============================================================================

static chpl_gpu_arena_t* device_arenas = NULL;
static size_t gpu_heap_size = 0;
static bool arena_allocator_enabled = false;  // disabled by default, enable via CHPL_RT_GPU_ARENA_ALLOCATOR


// this is compiler-generated
extern const char* chpl_gpuBinary;
extern const uint64_t chpl_gpuBinarySize;

static CUcontext *chpl_gpu_primary_ctx;
static CUdevice  *chpl_gpu_devices;

static int numAllDevices = -1;
static int numDevices = -1;
static int *dev_pid_to_lid_table;

// array indexed by device ID (we load the same module once for each GPU).
static CUmodule *chpl_gpu_cuda_modules;

static int *deviceClockRates;

static bool chpl_gpu_has_context(void) {
  CUcontext cuda_context = NULL;

  CUresult ret = cuCtxGetCurrent(&cuda_context);

  if (ret == CUDA_ERROR_NOT_INITIALIZED || ret == CUDA_ERROR_DEINITIALIZED) {
    return false;
  }
  else {
    return cuda_context != NULL;
  }
}

static void switch_context(int dev_lid) {
  CUcontext next_context = chpl_gpu_primary_ctx[dev_lid];

  if (!chpl_gpu_has_context()) {
    CUDA_CALL(cuCtxPushCurrent(next_context));
  }
  else {
    CUcontext cur_context = NULL;
    cuCtxGetCurrent(&cur_context);
    if (cur_context == NULL) {
      chpl_internal_error("Unexpected GPU context error");
    }

    if (cur_context != next_context) {
      CUcontext popped;
      CUDA_CALL(cuCtxPopCurrent(&popped));
      CUDA_CALL(cuCtxPushCurrent(next_context));
    }
  }
}

// Maps the "physical" device ID used by the CUDA library to the "logical"
// device ID used by this locale. The two may differ due to co-locales.
// Logical device IDs start with zero in each co-locale and are equal to the
// sublocale ID. Physical device IDs are the same for all co-locales on the
// machine.

static int dev_pid_to_lid(int32_t dev_pid) {
  assert((dev_pid >= 0) && (dev_pid < numAllDevices));
  int dev_lid = dev_pid_to_lid_table[dev_pid];
  assert((dev_lid >= 0) && (dev_lid < numDevices));
  return dev_lid;
}


static CUmodule get_module(void) {
  CUdevice device;
  CUmodule module;

  CUDA_CALL(cuCtxGetDevice(&device));
  int dev_lid = dev_pid_to_lid((int32_t) device);
  module = chpl_gpu_cuda_modules[dev_lid];
  return module;
}

extern c_nodeid_t chpl_nodeID;

// we can put this logic in chpl-gpu.c. However, it needs to execute
// per-context/module. That's currently too low level for that layer.
static void chpl_gpu_impl_set_globals(c_sublocid_t dev_lid, CUmodule module) {
  CUdeviceptr ptr;
  size_t glob_size;

  chpl_gpu_impl_load_global("chpl_nodeID", (void**)&ptr, &glob_size);

  assert(glob_size == sizeof(c_nodeid_t));
  chpl_gpu_impl_copy_host_to_device((void*)ptr, &chpl_nodeID, glob_size, NULL);
}

void chpl_gpu_impl_load_global(const char* global_name, void** ptr,
                               size_t* size) {
  CUmodule module = get_module();
  CUDA_CALL(cuModuleGetGlobal((CUdeviceptr*)ptr, size, module, global_name));
}

void* chpl_gpu_impl_load_function(const char* kernel_name) {
  CUfunction function;
  CUmodule module = get_module();
  CUDA_CALL(cuModuleGetFunction(&function, module, kernel_name));
  assert(function);

  return (void*)function;
}

void chpl_gpu_impl_use_device(c_sublocid_t dev_lid) {
  switch_context(dev_lid);
}

void chpl_gpu_impl_begin_init(int* num_all_devices) {
  CUDA_CALL(cuInit(0));
  CUDA_CALL(cuDeviceGetCount(&numAllDevices));
  *num_all_devices = numAllDevices;
}

void chpl_gpu_impl_collect_topo_addr_info(chpl_topo_pci_addr_t* into,
                                          int device_num) {
  CUdevice cuDevice;
  CUDA_CALL(cuDeviceGet(&cuDevice, device_num));
  int domain, bus, device;
  CUDA_CALL(cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                 cuDevice));
  CUDA_CALL(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                 cuDevice));
  CUDA_CALL(cuDeviceGetAttribute(&device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                 cuDevice));
  into->domain = (uint8_t) domain;
  into->bus = (uint8_t) bus;
  into->device = (uint8_t) device;
  into->function = 0;
}

// ============================================================================
// Arena Allocator Configuration from Environment
// ============================================================================

static void configure_arena_allocator(void) {
  // Allow enabling/disabling the arena allocator via environment variable
  // Default is false (disabled)
  arena_allocator_enabled = chpl_env_rt_get_bool("GPU_ARENA_ALLOCATOR", false);

  if (arena_allocator_enabled) {
    // Get heap size from environment variable
    // CHPL_RT_GPU_HEAP_SIZE supports size suffixes: k, K, m, M, g, G
    gpu_heap_size = chpl_env_rt_get_size("GPU_HEAP_SIZE", CHPL_GPU_DEFAULT_HEAP_SIZE);

    if (gpu_heap_size == 0) {
      arena_allocator_enabled = false;
    }
  }
}

// Allocate the GPU data structures. Note that the CUDA API, specifically
// cuCtxGetDevice, returns the global device ID so we need deviceIDToIndex
// to map from the global device ID to an array index.
void chpl_gpu_impl_setup_with_device_count(int num_my_devices) {
  numDevices = num_my_devices;
  chpl_gpu_primary_ctx = chpl_malloc(sizeof(CUcontext)*numDevices);
  chpl_gpu_devices = chpl_malloc(sizeof(CUdevice)*numDevices);
  chpl_gpu_cuda_modules = chpl_malloc(sizeof(CUmodule)*numDevices);
  deviceClockRates = chpl_malloc(sizeof(int)*numDevices);
  dev_pid_to_lid_table = chpl_malloc(sizeof(int) * numAllDevices);

  // Configure and allocate arena allocators (runtime-controlled)
  configure_arena_allocator();
  if (arena_allocator_enabled) {
    device_arenas = chpl_malloc(sizeof(chpl_gpu_arena_t) * numDevices);
    memset(device_arenas, 0, sizeof(chpl_gpu_arena_t) * numDevices);
  }
}

void chpl_gpu_impl_setup_device(int my_index, int global_index) {
  CUdevice device;
  CUDA_CALL(cuDeviceGet(&device, global_index));

  CUcontext context;

  CUDA_CALL(cuDevicePrimaryCtxSetFlags(device,
                                       CU_CTX_SCHED_BLOCKING_SYNC));
  CUDA_CALL(cuDevicePrimaryCtxRetain(&context, device));

  CUDA_CALL(cuCtxSetCurrent(context));
  // load the module and setup globals within
  CUmodule module = chpl_gpu_load_module(chpl_gpuBinary, chpl_gpuBinarySize);
  chpl_gpu_cuda_modules[my_index] = module;

  cuDeviceGetAttribute(&deviceClockRates[my_index],
                       CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);

  chpl_gpu_devices[my_index] = device;
  chpl_gpu_primary_ctx[my_index] = context;
  dev_pid_to_lid_table[global_index] = my_index; // map device ID to array index

  chpl_gpu_impl_set_globals(my_index, module);

  chpl_gpu_impl_setup_pgas();

  // Initialize arena allocator for this device (if enabled at runtime)
  if (arena_allocator_enabled && device_arenas != NULL) {
    CUdeviceptr arena_base = 0;
    CUresult result = cuMemAlloc(&arena_base, gpu_heap_size);
    if (result == CUDA_SUCCESS) {
      arena_init(&device_arenas[my_index], (void*)arena_base,
                 gpu_heap_size, my_index);
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "GPU arena allocation failed for device %d (requested %zu bytes). "
               "Set CHPL_RT_GPU_HEAP_SIZE to a smaller value or ensure sufficient GPU memory.",
               my_index, gpu_heap_size);
      chpl_internal_error(msg);
    }
  }
}

bool chpl_gpu_impl_stream_supported(void) {
  return true;
}

bool chpl_gpu_impl_is_device_ptr(const void* ptr) {
  return chpl_gpu_common_is_device_ptr(ptr);
}

bool chpl_gpu_impl_is_host_ptr(const void* ptr) {
  unsigned int res;
  CUresult ret_val = cuPointerGetAttribute(&res,
                                           CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                           (CUdeviceptr)ptr);

  if (ret_val != CUDA_SUCCESS) {
    if (ret_val == CUDA_ERROR_INVALID_VALUE ||
        ret_val == CUDA_ERROR_NOT_INITIALIZED ||
        ret_val == CUDA_ERROR_DEINITIALIZED) {
      return true;
    }
    else {
      CUDA_CALL(ret_val);
    }
  }
  else {
    return res == CU_MEMORYTYPE_HOST;
  }

  return true;
}

void chpl_gpu_impl_launch_kernel(void* kernel,
                                 int grd_dim_x, int grd_dim_y, int grd_dim_z,
                                 int blk_dim_x, int blk_dim_y, int blk_dim_z,
                                 void* stream, void** kernel_params) {
  assert(kernel);

  CUDA_CALL(cuLaunchKernel((CUfunction)kernel,
                           grd_dim_x, grd_dim_y, grd_dim_z,
                           blk_dim_x, blk_dim_y, blk_dim_z,
                           0,       // shared memory in bytes
                           (CUstream)stream,  // stream ID
                           kernel_params,
                           NULL));  // extra options
}

void* chpl_gpu_impl_memset(void* addr, const uint8_t val, size_t n,
                           void* stream) {
  assert(chpl_gpu_is_device_ptr(addr));

  CUDA_CALL(cuMemsetD8Async((CUdeviceptr)addr, (unsigned int)val, n,
                            (CUstream)stream));

  return addr;
}

void chpl_gpu_impl_copy_device_to_host(void* dst, const void* src, size_t n,
                                       void* stream) {
  assert(chpl_gpu_is_device_ptr(src));

  CUDA_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, n, (CUstream)stream));
}

void chpl_gpu_impl_copy_host_to_device(void* dst, const void* src, size_t n,
                                       void* stream) {
  assert(chpl_gpu_is_device_ptr(dst));

  CUDA_CALL(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, n, (CUstream)stream));
}

void chpl_gpu_impl_copy_device_to_device(void* dst, const void* src, size_t n,
                                         void* stream) {
  assert(chpl_gpu_is_device_ptr(dst) && chpl_gpu_is_device_ptr(src));

  CUDA_CALL(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, n,
                              (CUstream)stream))
}


void* chpl_gpu_impl_comm_async(void *dst, void *src, size_t n) {
  CUstream stream;
  cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, n, stream);
  return stream;
}

void chpl_gpu_impl_comm_wait(void *stream) {
  cuStreamSynchronize((CUstream)stream);
  cuStreamDestroy((CUstream)stream);
}

// ============================================================================
// Arena Allocator Helper Functions
// ============================================================================

// Get the arena for the current device (returns NULL if arena not available)
static chpl_gpu_arena_t* get_current_device_arena(void) {
  if (!arena_allocator_enabled || device_arenas == NULL) {
    return NULL;
  }

  CUdevice device;
  CUresult result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) {
    return NULL;
  }

  int dev_lid = dev_pid_to_lid((int32_t)device);
  if (dev_lid < 0 || dev_lid >= numDevices) {
    return NULL;
  }

  chpl_gpu_arena_t* arena = &device_arenas[dev_lid];
  if (!arena->initialized) {
    return NULL;
  }

  return arena;
}

// Get the arena that owns a pointer (returns NULL if not an arena pointer)
static chpl_gpu_arena_t* get_arena_for_ptr(const void* ptr) {
  if (!arena_allocator_enabled || device_arenas == NULL || ptr == NULL) {
    return NULL;
  }

  for (int i = 0; i < numDevices; i++) {
    if (device_arenas[i].initialized) {
      char* base = (char*)device_arenas[i].base;
      char* end = base + device_arenas[i].total_size;
      if ((char*)ptr >= base && (char*)ptr < end) {
        return &device_arenas[i];
      }
    }
  }
  return NULL;
}

// ============================================================================
// Memory Allocation Functions (using arena when available)
// ============================================================================

void* chpl_gpu_impl_mem_array_alloc(size_t size) {
  assert(size > 0);

  // Try arena allocator first (if enabled at runtime)
  chpl_gpu_arena_t* arena = get_current_device_arena();
  if (arena != NULL) {
    void* ptr = arena->ops->alloc(arena, size);
    if (ptr == NULL) {
      chpl_internal_error("GPU arena out of memory. "
                          "Increase CHPL_RT_GPU_HEAP_SIZE or reduce memory usage.");
    }
#if 0
    printf("Allocated %zu bytes from GPU arena at %p\n", size, ptr);
#endif
    return ptr;
  }

  // Fall back to standard CUDA allocation
#ifdef CHPL_GPU_MEM_STRATEGY_ARRAY_ON_DEVICE
  CUdeviceptr ptr = 0;
  CUDA_CALL(cuMemAlloc(&ptr, size));
  return (void*)ptr;
#else
  // For unified memory strategy, always use cuMemAllocManaged
  // since we need the managed memory semantics
  CUdeviceptr ptr = 0;
  CUDA_CALL(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));
  return (void*)ptr;
#endif
}


void* chpl_gpu_impl_mem_alloc(size_t size) {
  //// Do not use the arena for non-array allocations for now
  // Try arena allocator first (if enabled at runtime)
  // chpl_gpu_arena_t* arena = get_current_device_arena();
  // if (arena != NULL) {
  //   void* ptr = arena->ops->alloc(arena, size);
  //   if (ptr == NULL) {
  //     chpl_internal_error("GPU arena out of memory. "
  //                         "Increase CHPL_RT_GPU_HEAP_SIZE or reduce memory usage.");
  //   }
  //   return ptr;
  // }

  // Fall back to standard CUDA allocation
#ifdef CHPL_GPU_MEM_STRATEGY_ARRAY_ON_DEVICE
  // For non-array allocations in ARRAY_ON_DEVICE mode,
  // use host-pinned memory (not arena)
  void* ptr = 0;
  CUDA_CALL(cuMemAllocHost(&ptr, size));
  assert(ptr != 0);
  return (void*)ptr;
#else
  CUdeviceptr ptr = 0;
  CUDA_CALL(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));
  assert(ptr != 0);
  return (void*)ptr;
#endif
}

void chpl_gpu_impl_mem_free(void* memAlloc) {
  if (memAlloc != NULL) {
    // Check if this is an arena pointer (runtime check)
    chpl_gpu_arena_t* arena = get_arena_for_ptr(memAlloc);
    if (arena != NULL) {
      arena->ops->free(arena, memAlloc);
      return;
    }

    // Not an arena pointer, use regular CUDA free
    assert(chpl_gpu_is_device_ptr(memAlloc));

    // see note in chpl_gpu_mem_free
    int32_t dev_pid = -1;
    CUDA_CALL(cuPointerGetAttribute((void*)&dev_pid,
                                    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                    (CUdeviceptr)memAlloc));
    if (dev_pid != -1) {
      int dev_lid = dev_pid_to_lid(dev_pid);
      switch_context(dev_lid);
    }

#ifdef CHPL_GPU_MEM_STRATEGY_ARRAY_ON_DEVICE
    if (chpl_gpu_impl_is_host_ptr(memAlloc)) {
      CUDA_CALL(cuMemFreeHost(memAlloc));
    }
    else {
      CUDA_CALL(cuMemFree((CUdeviceptr)memAlloc));
    }
#else
    CUDA_CALL(cuMemFree((CUdeviceptr)memAlloc));
#endif
  }
}

void chpl_gpu_impl_hostmem_register(void *memAlloc, size_t size) {
  // The CUDA driver uses DMA to transfer page-locked memory to the GPU; if
  // memory is not page-locked it must first be transferred into a page-locked
  // buffer, which degrades performance. So in the array_on_device mode we
  // choose to page-lock such memory even if it's on the host-side.
  #ifdef CHPL_GPU_MEM_STRATEGY_ARRAY_ON_DEVICE
  cudaHostRegister(memAlloc, size, cudaHostRegisterPortable);
  #endif
}

// This can be used for proper reallocation
size_t chpl_gpu_impl_get_alloc_size(void* ptr) {
  // Check if this is an arena pointer (runtime check)
  chpl_gpu_arena_t* arena = get_arena_for_ptr(ptr);
  if (arena != NULL) {
    return arena->ops->get_alloc_size(arena, ptr);
  }

  // Not an arena pointer, use CUDA API
  return chpl_gpu_common_get_alloc_size(ptr);
}

unsigned int chpl_gpu_device_clock_rate(int32_t devNum) {
  return (unsigned int)deviceClockRates[devNum];
}

void* chpl_gpu_get_module(void) {
  CUmodule module = get_module();
  if (module == NULL) {
    chpl_internal_error("chpl_gpu_get_module: no module loaded");
  }
  return (void*)module;
}

bool chpl_gpu_impl_can_access_peer(int dev_lid1, int dev_lid2) {
  int p2p;
  CUDA_CALL(cuDeviceCanAccessPeer(&p2p, chpl_gpu_devices[dev_lid1],
    chpl_gpu_devices[dev_lid2]));
  return p2p != 0;
}

void chpl_gpu_impl_set_peer_access(int dev_lid1, int dev_lid2, bool enable) {
  switch_context(dev_lid1);
  if(enable) {
    CUDA_CALL(cuCtxEnablePeerAccess(chpl_gpu_primary_ctx[dev_lid2], 0));
  } else {
    CUDA_CALL(cuCtxDisablePeerAccess(chpl_gpu_primary_ctx[dev_lid2]));
  }
}

void chpl_gpu_impl_synchronize(void) {
  CUDA_CALL(cuCtxSynchronize());
}

void* chpl_gpu_impl_stream_create(void) {
  CUstream stream;
  CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  return (void*) stream;
}

void chpl_gpu_impl_stream_destroy(void* stream) {
  if (stream) {
    CUDA_CALL(cuStreamDestroy((CUstream)stream));
  }
}

bool chpl_gpu_impl_stream_ready(void* stream) {
  if (stream) {
    CUresult res = cuStreamQuery(stream);
    if (res == CUDA_ERROR_NOT_READY) {
      return false;
    }
    CUDA_CALL(res);
  }
  return true;
}

void chpl_gpu_impl_stream_synchronize(void* stream) {
  if (stream) {
    CUDA_CALL(cuStreamSynchronize(stream));
  }
}

void* chpl_gpu_impl_host_register(void* var, size_t size) {
  CUDA_CALL(cuMemHostRegister(var, size, CU_MEMHOSTREGISTER_PORTABLE));
  return var;
}

void chpl_gpu_impl_host_unregister(void* var) {
  CUDA_CALL(cuMemHostUnregister(var));
}

void chpl_gpu_impl_name(int dev, char *resultBuffer, int bufferSize) {
  CUDA_CALL(cuDeviceGetName(resultBuffer, bufferSize, chpl_gpu_devices[dev]));
}

const int CHPL_GPU_ATTRIBUTE__MAX_THREADS_PER_BLOCK = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
const int CHPL_GPU_ATTRIBUTE__MAX_BLOCK_DIM_X = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
const int CHPL_GPU_ATTRIBUTE__MAX_BLOCK_DIM_Y = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
const int CHPL_GPU_ATTRIBUTE__MAX_BLOCK_DIM_Z = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
const int CHPL_GPU_ATTRIBUTE__MAX_GRID_DIM_X = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
const int CHPL_GPU_ATTRIBUTE__MAX_GRID_DIM_Y = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
const int CHPL_GPU_ATTRIBUTE__MAX_GRID_DIM_Z = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
const int CHPL_GPU_ATTRIBUTE__MAX_SHARED_MEMORY_PER_BLOCK = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
const int CHPL_GPU_ATTRIBUTE__TOTAL_CONSTANT_MEMORY = CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
const int CHPL_GPU_ATTRIBUTE__WARP_SIZE = CU_DEVICE_ATTRIBUTE_WARP_SIZE;
const int CHPL_GPU_ATTRIBUTE__MAX_PITCH = CU_DEVICE_ATTRIBUTE_MAX_PITCH;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE1D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE2D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE2D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE3D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE3D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT;
const int CHPL_GPU_ATTRIBUTE__MAXIMUM_TEXTURE3D_DEPTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH;
const int CHPL_GPU_ATTRIBUTE__MAX_REGISTERS_PER_BLOCK = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
const int CHPL_GPU_ATTRIBUTE__CLOCK_RATE = CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
const int CHPL_GPU_ATTRIBUTE__TEXTURE_ALIGNMENT = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
const int CHPL_GPU_ATTRIBUTE__TEXTURE_PITCH_ALIGNMENT = CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT;
const int CHPL_GPU_ATTRIBUTE__MULTIPROCESSOR_COUNT = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
const int CHPL_GPU_ATTRIBUTE__KERNEL_EXEC_TIMEOUT = CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT;
const int CHPL_GPU_ATTRIBUTE__INTEGRATED = CU_DEVICE_ATTRIBUTE_INTEGRATED;
const int CHPL_GPU_ATTRIBUTE__CAN_MAP_HOST_MEMORY = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
const int CHPL_GPU_ATTRIBUTE__COMPUTE_MODE = CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
const int CHPL_GPU_ATTRIBUTE__CONCURRENT_KERNELS = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
const int CHPL_GPU_ATTRIBUTE__ECC_ENABLED = CU_DEVICE_ATTRIBUTE_ECC_ENABLED;
const int CHPL_GPU_ATTRIBUTE__PCI_BUS_ID = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
const int CHPL_GPU_ATTRIBUTE__PCI_DEVICE_ID = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
const int CHPL_GPU_ATTRIBUTE__MEMORY_CLOCK_RATE = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
const int CHPL_GPU_ATTRIBUTE__GLOBAL_MEMORY_BUS_WIDTH = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
const int CHPL_GPU_ATTRIBUTE__L2_CACHE_SIZE = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
const int CHPL_GPU_ATTRIBUTE__MAX_THREADS_PER_MULTIPROCESSOR = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
const int CHPL_GPU_ATTRIBUTE__COMPUTE_CAPABILITY_MAJOR = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
const int CHPL_GPU_ATTRIBUTE__COMPUTE_CAPABILITY_MINOR = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
const int CHPL_GPU_ATTRIBUTE__MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR;
const int CHPL_GPU_ATTRIBUTE__MANAGED_MEMORY = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
const int CHPL_GPU_ATTRIBUTE__MULTI_GPU_BOARD = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD;
const int CHPL_GPU_ATTRIBUTE__PAGEABLE_MEMORY_ACCESS = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
const int CHPL_GPU_ATTRIBUTE__CONCURRENT_MANAGED_ACCESS = CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
const int CHPL_GPU_ATTRIBUTE__PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
const int CHPL_GPU_ATTRIBUTE__DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST;

int chpl_gpu_impl_query_attribute(int dev, int attribute) {
  int res;
  CUDA_CALL(cuDeviceGetAttribute(&res, attribute, chpl_gpu_devices[dev]));
  return res;
}

#endif // HAS_GPU_LOCALE
