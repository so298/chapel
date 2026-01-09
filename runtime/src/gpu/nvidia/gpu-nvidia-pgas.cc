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

#include "chpl-comm.h"
#include "chpl-env-gen.h"
#include "chpl-env.h"
#include "chpl-gpu-impl.h"
#include "chpl-gpu.h"
#include "chpl-linefile-support.h"
#include "chpl-mem.h"
#include "chpl-tasks.h"
#include "chpl-topo.h"
#include "chplcgfns.h"
#include "chplrt.h"
#include "error.h"
#include "sys_basic.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
// or other algorithms.
//
// To enable this allocator, call chpl_gpu_impl_pgas_enable(1) before
// chpl_gpu_arena_configure(). The heap size can be configured via
// chpl_gpu_impl_pgas_set_heap_size() (default 1GiB).
// ============================================================================

namespace {

// Default heap size: 1 GiB
constexpr size_t kDefaultHeapSize = size_t{1} << 30;

// Minimum allocation alignment (16 bytes for GPU efficiency)
constexpr size_t kAlignment = 16;

// Minimum block size
constexpr size_t kMinBlockSize = 64;

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

struct ArenaBlock {
  size_t offset;         // Offset from arena base in device memory
  size_t size;           // Size of this block
  bool allocated;        // Whether this block is allocated
  ArenaBlock *next;      // Next block by offset order
  ArenaBlock *prev;      // Previous block by offset order
  ArenaBlock *next_free; // Next free block (only valid if free)
  ArenaBlock *prev_free; // Previous free block (only valid if free)

  ArenaBlock(size_t off, size_t sz)
      : offset(off), size(sz), allocated(false), next(nullptr), prev(nullptr),
        next_free(nullptr), prev_free(nullptr) {}

  static ArenaBlock *create(size_t offset, size_t size) {
    void *mem =
        chpl_mem_alloc(sizeof(ArenaBlock), CHPL_RT_MD_GPU_KERNEL_ARG, 0, 0);
    return new (mem) ArenaBlock(offset, size);
  }

  static void destroy(ArenaBlock *block) {
    block->~ArenaBlock();
    chpl_mem_free(block, 0, 0);
  }
};

// ============================================================================
// GPU Arena Class
// ============================================================================
// Manages a single arena for one GPU device.

class GpuArena {
public:
  GpuArena()
      : base_(nullptr), total_size_(0), used_size_(0), block_list_(nullptr),
        free_list_(nullptr), device_id_(-1), initialized_(false) {}

  ~GpuArena() { destroy(); }

  void init(void *base, size_t size, int device_id) {
    base_ = base;
    total_size_ = size;
    used_size_ = 0;
    device_id_ = device_id;
    initialized_ = true;

    // Initialize the entire arena as one big free block
    ArenaBlock *initial_block = ArenaBlock::create(0, size);
    block_list_ = initial_block;
    free_list_ = initial_block;
  }

  void destroy() {
    if (!initialized_)
      return;

    // Free all block metadata
    ArenaBlock *block = block_list_;
    while (block != nullptr) {
      ArenaBlock *next = block->next;
      ArenaBlock::destroy(block);
      block = next;
    }
    block_list_ = nullptr;
    free_list_ = nullptr;
    initialized_ = false;
  }

  void *alloc(size_t size) {
    // Align size and ensure minimum
    size_t aligned_size = alignUp(size);
    if (aligned_size < kMinBlockSize) {
      aligned_size = kMinBlockSize;
    }

    // First-fit search through free list
    ArenaBlock *block = free_list_;
    while (block != nullptr) {
      assert(!block->allocated);

      if (block->size >= aligned_size) {
        // Found a suitable block
        size_t remaining = block->size - aligned_size;

        // Check if we can split this block
        if (remaining >= kMinBlockSize) {
          splitBlock(block, aligned_size, remaining);
        } else {
          // Use the whole block, remove from free list
          removeFromFreeList(block);
        }

        // Mark as allocated
        block->allocated = true;
        used_size_ += block->size;

        return offsetToPtr(block->offset);
      }

      block = block->next_free;
    }

    // No suitable block found
    return nullptr;
  }

  void free(void *ptr) {
    if (ptr == nullptr)
      return;

    // Find the block for this pointer
    ArenaBlock *block = findBlockByPtr(ptr);
    if (block == nullptr) {
      chpl_internal_error(
          "GPU arena: invalid free - pointer not found in arena");
    }
    if (!block->allocated) {
      chpl_internal_error("GPU arena: double free detected");
    }

    // Mark as free
    block->allocated = false;
    used_size_ -= block->size;

    // Coalesce with adjacent free blocks
    coalesceWithNext(block);
    block = coalesceWithPrev(block);

    // Add to free list
    insertToFreeList(block);
  }

  size_t getAllocSize(void *ptr) const {
    if (ptr == nullptr)
      return 0;

    ArenaBlock *block = findBlockByPtr(ptr);
    if (block == nullptr || !block->allocated) {
      return 0;
    }
    return block->size;
  }

  bool containsPtr(const void *ptr) const {
    if (!initialized_ || ptr == nullptr)
      return false;
    const char *p = static_cast<const char *>(ptr);
    const char *base = static_cast<const char *>(base_);
    return p >= base && p < base + total_size_;
  }

  bool isInitialized() const { return initialized_; }
  void *base() const { return base_; }
  size_t totalSize() const { return total_size_; }
  size_t usedSize() const { return used_size_; }
  int deviceId() const { return device_id_; }

private:
  static size_t alignUp(size_t size) {
    return (size + kAlignment - 1) & ~(kAlignment - 1);
  }

  void *offsetToPtr(size_t offset) const {
    return static_cast<char *>(base_) + offset;
  }

  size_t ptrToOffset(const void *ptr) const {
    return static_cast<size_t>(static_cast<const char *>(ptr) -
                               static_cast<const char *>(base_));
  }

  ArenaBlock *findBlockByPtr(const void *ptr) const {
    size_t target_offset = ptrToOffset(ptr);
    ArenaBlock *block = block_list_;
    while (block != nullptr) {
      if (block->offset == target_offset) {
        return block;
      }
      block = block->next;
    }
    return nullptr;
  }

  void removeFromFreeList(ArenaBlock *block) {
    if (block->prev_free) {
      block->prev_free->next_free = block->next_free;
    } else {
      free_list_ = block->next_free;
    }
    if (block->next_free) {
      block->next_free->prev_free = block->prev_free;
    }
    block->next_free = nullptr;
    block->prev_free = nullptr;
  }

  void insertToFreeList(ArenaBlock *block) {
    block->next_free = free_list_;
    block->prev_free = nullptr;
    if (free_list_) {
      free_list_->prev_free = block;
    }
    free_list_ = block;
  }

  void splitBlock(ArenaBlock *block, size_t aligned_size, size_t remaining) {
    // Create a new block for the remaining space
    ArenaBlock *new_block =
        ArenaBlock::create(block->offset + aligned_size, remaining);

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
      free_list_ = new_block;
    }
    if (new_block->next_free) {
      new_block->next_free->prev_free = new_block;
    }
    block->next_free = nullptr;
    block->prev_free = nullptr;
  }

  void coalesceWithNext(ArenaBlock *block) {
    if (block->next != nullptr && !block->next->allocated) {
      ArenaBlock *next = block->next;

      // Remove next from free list
      removeFromFreeList(next);

      // Merge next into block
      block->size += next->size;
      block->next = next->next;
      if (next->next) {
        next->next->prev = block;
      }

      // Free the merged block's metadata
      ArenaBlock::destroy(next);
    }
  }

  ArenaBlock *coalesceWithPrev(ArenaBlock *block) {
    if (block->prev != nullptr && !block->prev->allocated) {
      ArenaBlock *prev = block->prev;

      // Remove prev from free list
      removeFromFreeList(prev);

      // Merge block into prev
      prev->size += block->size;
      prev->next = block->next;
      if (block->next) {
        block->next->prev = prev;
      }

      // Free block's metadata and return prev instead
      ArenaBlock::destroy(block);
      return prev;
    }
    return block;
  }

  void *base_;
  size_t total_size_;
  size_t used_size_;
  ArenaBlock *block_list_;
  ArenaBlock *free_list_;
  int device_id_;
  bool initialized_;
};

// ============================================================================
// GPU Arena Manager (Singleton)
// ============================================================================
// Manages arenas for all GPU devices.

class GpuArenaManager {
public:
  static GpuArenaManager &instance() {
    static GpuArenaManager mgr;
    return mgr;
  }

  // Configure and initialize arena structures for all devices.
  void configure(int num_devices) {
    // Allocate arena structures for all devices
    num_devices_ = num_devices;
    arenas_ = static_cast<GpuArena *>(chpl_mem_alloc(
        sizeof(GpuArena) * num_devices, CHPL_RT_MD_GPU_KERNEL_ARG, 0, 0));
    // Placement new for each arena
    for (int i = 0; i < num_devices; i++) {
      new (&arenas_[i]) GpuArena();
    }
  }

  void initDeviceArena(int device_index, void *base, size_t size) {
    if (arenas_ == nullptr)
      return;
    if (base == nullptr || size == 0)
      return;

    arenas_[device_index].init(base, size, device_index);
  }

  GpuArena *getCurrentDeviceArena(int device_lid) {
    if (arenas_ == nullptr)
      return nullptr;
    if (device_lid < 0 || device_lid >= num_devices_)
      return nullptr;
    if (!arenas_[device_lid].isInitialized())
      return nullptr;
    return &arenas_[device_lid];
  }

  GpuArena *getArenaForPtr(const void *ptr) {
    if (arenas_ == nullptr || ptr == nullptr)
      return nullptr;

    for (int i = 0; i < num_devices_; i++) {
      if (arenas_[i].containsPtr(ptr)) {
        return &arenas_[i];
      }
    }
    return nullptr;
  }

  bool isInitialized() const { return arenas_ != nullptr; }

private:
  GpuArenaManager() : arenas_(nullptr), num_devices_(0) {}

  ~GpuArenaManager() {
    if (arenas_ != nullptr) {
      for (int i = 0; i < num_devices_; i++) {
        arenas_[i].~GpuArena();
      }
      chpl_mem_free(arenas_, 0, 0);
      arenas_ = nullptr;
    }
  }

  GpuArenaManager(const GpuArenaManager &) = delete;
  GpuArenaManager &operator=(const GpuArenaManager &) = delete;

  GpuArena *arenas_;
  int num_devices_;
};

} // anonymous namespace

// ============================================================================
// Extern "C" API for Arena Allocator
// ============================================================================

extern "C" {

void chpl_gpu_arena_configure(int num_devices) {
  GpuArenaManager::instance().configure(num_devices);
}

int chpl_gpu_arena_enabled(void) { return chpl_gpu_impl_pgas_is_enabled(); }

// Initialize the arena for a specific device with pre-allocated GPU memory.
// The caller is responsible for allocating the GPU memory (e.g., via
// nvshmem_malloc) and passing the base pointer and size here.
void chpl_gpu_arena_init_device(int device_index, void *base, size_t size) {
  if (GpuArenaManager::instance().isInitialized() == false) {
    // n_devices is just 1 for now
    GpuArenaManager::instance().configure(1);
  }
  GpuArenaManager::instance().initDeviceArena(device_index, base, size);
}

void *chpl_gpu_arena_alloc(int device_lid, size_t size) {
  GpuArena *arena =
      GpuArenaManager::instance().getCurrentDeviceArena(device_lid);
  if (arena == nullptr) {
    printf("chpl_gpu_arena_alloc: Arena not initialized for device %d\n",
           device_lid);
    return nullptr;
  }

  void *ptr = arena->alloc(size);
  if (ptr == nullptr) {
    chpl_internal_error(
        "GPU arena out of memory. "
        "Increase CHPL_RT_GPU_HEAP_SIZE or reduce memory usage.");
  }
  return ptr;
}

void chpl_gpu_arena_free(void *ptr) {
  if (ptr == nullptr)
    return;

  GpuArena *arena = GpuArenaManager::instance().getArenaForPtr(ptr);
  if (arena != nullptr) {
    arena->free(ptr);
  }
}

size_t chpl_gpu_arena_get_alloc_size(void *ptr) {
  if (ptr == nullptr)
    return 0;

  GpuArena *arena = GpuArenaManager::instance().getArenaForPtr(ptr);
  if (arena != nullptr) {
    return arena->getAllocSize(ptr);
  }
  return 0;
}

int chpl_gpu_arena_is_ptr(const void *ptr) {
  return GpuArenaManager::instance().getArenaForPtr(ptr) != nullptr ? 1 : 0;
}

} // extern "C"

// ============================================================================
// GPU PGAS Implementation
// ============================================================================

static int pgas_enabled_ = 0;

// Enable or disable GPU PGAS support.
// Must be called before chpl_gpu_arena_configure().
void chpl_gpu_impl_pgas_enable(int enable) { pgas_enabled_ = enable ? 1 : 0; }

// Returns whether GPU PGAS is enabled.
int chpl_gpu_impl_pgas_is_enabled(void) { return pgas_enabled_; }

void chpl_gpu_pgas_get_uid(u_int8_t uid_out[128]) {
  printf("chpl_gpu_pgas_get_uid: Getting NVSHMEM unique ID\n");
  nvshmemx_uniqueid_t *uid_ptr = (nvshmemx_uniqueid_t *)uid_out;
  *uid_ptr = NVSHMEMX_UNIQUEID_INITIALIZER;
  int status;
  status = nvshmemx_get_uniqueid(uid_ptr);
  if (status != 0) {
    fprintf(stderr, "[%s] Failed to get NVSHMEM unique ID, status: %d\n",
            __func__, status);
    exit(1);
  }
}

void chpl_gpu_pgas_init_with_uid(int rank, int nranks, u_int8_t uid_in[128]) {
  printf("chpl_gpu_pgas_init_with_uid: Initializing NVSHMEM on rank %d/%d\n",
         rank, nranks);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

  nvshmemx_set_attr_uniqueid_args(rank, nranks, (nvshmemx_uniqueid_t *)uid_in,
                                  &attr);

  printf("chpl_gpu_pgas_init_with_uid: Calling nvshmemx_hostlib_init_attr\n");

  int status;
  // status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  printf("chpl_gpu_pgas_init_with_uid: NVSHMEM initialized on rank %d/%d\n",
         rank, nranks);
  if (status != 0) {
    fprintf(stderr,
            "[%s] Failed to initialize NVSHMEM with unique ID, status: %d\n",
            __func__, status);
    exit(1);
  }

  chpl_gpu_impl_pgas_enable(1);
}

void chpl_gpu_pgas_setup_heap_arena(size_t pgas_heap_size) {
  // int num_devices = chpl_gpu_getNumDevices();
  const int num_devices = 1; // For now, assume single GPU per rank

  printf("chpl_gpu_pgas_setup_heap_arena: Setting up GPU PGAS heap arena "
         "of size %zu bytes on %d devices\n",
         pgas_heap_size, num_devices);

  // Allocate PGAS heap arena on each device
  for (int dev = 0; dev < num_devices; dev++) {
    chpl_gpu_impl_use_device(dev);

    void *pgas_heap_base = nvshmem_malloc(pgas_heap_size);
    if (pgas_heap_base == nullptr) {
      chpl_internal_error("Failed to allocate NVSHMEM PGAS heap on device");
    }

    printf("chpl_gpu_pgas_setup_heap_arena: Device %d PGAS heap base: %p\n",
           dev, pgas_heap_base);

    // Initialize the GPU arena with the allocated PGAS heap
    chpl_gpu_arena_init_device(dev, pgas_heap_base, pgas_heap_size);
  }
}

void *chpl_gpu_pgas_sym_malloc(const size_t size) {
  return nvshmem_malloc(size);
}

void chpl_gpu_pgas_sym_free(void *ptr) { nvshmem_free(ptr); }

__global__ void gpu_pgas_get_kernel(void *dst, void *src, size_t size,
                                    int node) {
  nvshmem_getmem(dst, src, size, node);
}

void chpl_gpu_impl_pgas_comm_get(void *dst, c_nodeid_t node, void *src,
                                 size_t size) {
#ifdef RECORD_TIME
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  // auto ptr = nvshmem_ptr(dst, (int)node);
  // if (ptr == NULL) {
  //   printf("chpl_gpu_impl_pgas_comm_put: nvshmem_ptr returned null\n");
  // }
  // auto ptr2 = nvshmem_ptr(src, nvshmem_my_pe());
  // if (ptr2 == NULL) {
  //   printf("chpl_gpu_impl_pgas_comm_put: nvshmem_ptr returned null for
  //   src\n");
  // }

  cudaStream_t stream;
  cudaError_t cuda_status = cudaStreamCreate(&stream);
  if (cuda_status != cudaSuccess) {
    chpl_internal_error("Failed to create CUDA stream for PGAS comm get");
  }
  // printf("chpl_gpu_impl_pgas_comm_get: node=%d size=%zu\n", (int)node, size);
  // nvshmemx_getmem_on_stream(dst, src, size, (int)node, stream);
  gpu_pgas_get_kernel<<<1, 1, 0, stream>>>(dst, src, size, (int)node);
  cudaDeviceSynchronize();
  // printf("chpl_gpu_impl_pgas_comm_get: after nvshmem_getmem\n");
  // nvshmem_quiet();

  // Destroy the stream
  cuda_status = cudaStreamDestroy(stream);
  if (cuda_status != cudaSuccess) {
    chpl_internal_error("Failed to destroy CUDA stream for PGAS comm get");
  }

#ifdef RECORD_TIME
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  printf("chpl_gpu_impl_pgas_comm_get: node=%d size=%zu time=%ld us\n",
         (int)node, size, duration.count());
#endif
}

__global__ void gpu_pgas_put_kernel(void *dst, void *src, size_t size,
                                    int node) {
  nvshmem_putmem(dst, src, size, node);
}

void chpl_gpu_impl_pgas_comm_put(void *dst, c_nodeid_t node, void *src,
                                 size_t size) {
#ifdef RECORD_TIME
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  // Create a stream for the operation
  cudaStream_t stream;
  cudaError_t cuda_status = cudaStreamCreate(&stream);
  if (cuda_status != cudaSuccess) {
    chpl_internal_error("Failed to create CUDA stream for PGAS comm put");
  }

  // printf("chpl_gpu_impl_pgas_comm_put: node=%d size=%zu\n", (int)node, size);
  // nvshmemx_putmem_on_stream(dst, src, size, (int)node, stream);
  gpu_pgas_put_kernel<<<1, 1, 0, stream>>>(dst, src, size, (int)node);
  cudaDeviceSynchronize();
  // printf("chpl_gpu_impl_pgas_comm_put: after nvshmem_putmem\n");
  // nvshmem_quiet();

  // Destroy the stream
  cuda_status = cudaStreamDestroy(stream);
  if (cuda_status != cudaSuccess) {
    chpl_internal_error("Failed to destroy CUDA stream for PGAS comm put");
  }

#ifdef RECORD_TIME
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  printf("chpl_gpu_impl_pgas_comm_put: node=%d size=%zu time=%ld us\n",
         (int)node, size, duration.count());
#endif
}
