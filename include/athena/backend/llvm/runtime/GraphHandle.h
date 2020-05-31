#include <athena/backend/llvm/runtime/Device.h>

#include <memory>
#include <vector>

#pragma once

namespace athena::backend::llvm {
class BackendAllocator;
}

struct GraphHandle {
  std::shared_ptr<athena::backend::llvm::BackendAllocator> allocator;
  std::vector<athena::backend::llvm::Device*> devices;
};
