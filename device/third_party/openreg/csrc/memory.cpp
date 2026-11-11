#include "memory.h"

#include <include/openreg.h>

#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>

namespace {

struct Block {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
  int refcount{0};
  uint64_t sim_addr = 0;
  uint64_t sim_size = 0;
};

// The simulated DRAM device blocks are allocated in: 16 KiB-aligned ranges from 0, first fit over the holes that
// freed blocks left, else above everything allocated; a freed range coalesces with its neighbours.
class SimDram {
 public:
  static constexpr uint64_t kAlign = 16 * 1024;

  uint64_t allocate(uint64_t bytes) {
    const uint64_t need = (bytes + kAlign - 1) / kAlign * kAlign;
    for (auto it = m_holes.begin(); it != m_holes.end(); ++it) {
      if (it->second < need)
        continue;
      const uint64_t at = it->first, left = it->second - need;
      m_holes.erase(it);
      if (left)
        m_holes[at + need] = left;
      return at;
    }
    const uint64_t at = m_top;
    m_top += need;
    return at;
  }

  void free(uint64_t at, uint64_t bytes) {
    uint64_t start = at, len = (bytes + kAlign - 1) / kAlign * kAlign;
    auto next = m_holes.lower_bound(start);
    if (next != m_holes.end() && start + len == next->first) {
      len += next->second;
      next = m_holes.erase(next);
    }
    if (next != m_holes.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == start) {
        start = prev->first;
        len += prev->second;
        m_holes.erase(prev);
      }
    }
    if (start + len == m_top)
      m_top = start;
    else
      m_holes[start] = len;
  }

 private:
  std::map<uint64_t, uint64_t> m_holes;
  uint64_t m_top = 0;
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  orError_t allocate(void** ptr, size_t size, orMemoryType type) {
    if (!ptr || size == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    long page_size = openreg::get_pagesize();
    size_t aligned_size = ((size - 1) / page_size + 1) * page_size;
    void* mem = nullptr;
    int current_device = -1;

    if (type == orMemoryType::orMemoryTypeDevice) {
      orGetDevice(&current_device);

      mem = openreg::mmap(aligned_size);
      if (mem == nullptr)
        return orErrorUnknown;
      if (openreg::mprotect(mem, aligned_size, F_PROT_NONE) != 0) {
        openreg::munmap(mem, aligned_size);
        return orErrorUnknown;
      }
    } else {
      if (openreg::alloc(&mem, page_size, aligned_size) != 0) {
        return orErrorUnknown;
      }
    }

    Block block{type, current_device, mem, aligned_size, 0};
    if (type == orMemoryType::orMemoryTypeDevice) {
      block.sim_size = size;
      block.sim_addr = m_sim_dram.allocate(size);
    }
    m_registry[mem] = block;
    *ptr = mem;
    return orSuccess;
  }

  orError_t free(void* ptr) {
    if (!ptr)
      return orSuccess;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return orErrorUnknown;

    const auto& info = it->second;
    if (info.type == orMemoryType::orMemoryTypeDevice) {
      m_sim_dram.free(info.sim_addr, info.sim_size);
      openreg::mprotect(info.pointer, info.size, F_PROT_READ | F_PROT_WRITE);
      openreg::munmap(info.pointer, info.size);
    } else {
      openreg::free(info.pointer);
    }

    m_registry.erase(it);
    return orSuccess;
  }

  orError_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      orMemcpyKind kind) {
    if (!dst || !src || count == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* dst_info = getBlockInfoNoLock(dst);
    Block* src_info = getBlockInfoNoLock(src);

    switch (kind) {
      case orMemcpyHostToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyHostToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
    }

    unprotectNoLock(dst_info);
    unprotectNoLock(src_info);
    ::memcpy(dst, src, count);
    protectNoLock(dst_info);
    protectNoLock(src_info);

    return orSuccess;
  }

  orError_t getPointerAttributes(
      orPointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return orErrorUnknown;

    std ::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = orMemoryType::orMemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return orSuccess;
  }

  orError_t getSimAddress(uint64_t* addr, const void* ptr) {
    if (!addr || !ptr)
      return orErrorUnknown;
    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);
    if (!info || info->type != orMemoryType::orMemoryTypeDevice)
      return orErrorUnknown;
    *addr = info->sim_addr + static_cast<uint64_t>(static_cast<const char*>(ptr) -
                                                   static_cast<const char*>(info->pointer));
    return orSuccess;
  }

  orError_t unprotect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return unprotectNoLock(getBlockInfoNoLock(ptr));
  }

  orError_t protect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return protectNoLock(getBlockInfoNoLock(ptr));
  }

 private:
  MemoryManager() = default;

  orError_t unprotectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 0) {
        if (openreg::mprotect(
                info->pointer, info->size, F_PROT_READ | F_PROT_WRITE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount++;
    }

    return orSuccess;
  }

  orError_t protectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 1) {
        if (openreg::mprotect(info->pointer, info->size, F_PROT_NONE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount--;
    }

    return orSuccess;
  }

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
        return &it->second;
      }
    }

    return nullptr;
  }

  std::map<void*, Block> m_registry;
  SimDram m_sim_dram;
  std::mutex m_mutex;
};

} // namespace

orError_t orMalloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, orMemoryType::orMemoryTypeDevice);
}

orError_t orFree(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

orError_t orMallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, orMemoryType::orMemoryTypeHost);
}

orError_t orFreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

orError_t orMemcpy(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  auto& mm = MemoryManager::getInstance();

  return orLaunchKernel(
      stream, &MemoryManager::memcpy, &mm, dst, src, count, kind);
}

orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

orError_t orPointerGetSimAddress(uint64_t* simAddr, const void* devPtr) {
  return MemoryManager::getInstance().getSimAddress(simAddr, devPtr);
}

size_t orSimAddressAlignment(void) {
  return SimDram::kAlign;
}

orError_t orMemoryUnprotect(void* devPtr) {
  return MemoryManager::getInstance().unprotect(devPtr);
}

orError_t orMemoryProtect(void* devPtr) {
  return MemoryManager::getInstance().protect(devPtr);
}
