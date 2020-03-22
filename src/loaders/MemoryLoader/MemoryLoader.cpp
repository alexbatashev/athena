/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/loaders/MemoryLoader/MemoryLoader.h>

#include <cstring>

namespace athena::loaders {

MemoryLoader::MemoryLoader(MemoryLoader&& src) noexcept {
  this->mData = src.mData;
  this->mSize = src.mSize;
  src.mSize = 0;
  src.mData = nullptr;
}

MemoryLoader&
MemoryLoader::operator=(athena::loaders::MemoryLoader&& src) noexcept {
  if (&src == this) {
    return *this;
  }

  this->mData = src.mData;
  this->mSize = src.mSize;
  src.mSize = 0;
  src.mData = nullptr;

  return *this;
}

void MemoryLoader::load(core::Allocator* allocator,
                        core::inner::Tensor* tensor) {
  allocator->lock(*tensor);
  auto pointer = allocator->get(*tensor);

  athena_assert(pointer, "MemoryLoader pointer is NULL");
  athena_assert(mSize <= tensor->getShapeView().getTotalSize() *
                             core::sizeOfDataType(tensor->getDataType()),
                "Size is greater than tensor size.");
  std::memmove(pointer, mData, mSize);
  allocator->release(*tensor);
}
std::string MemoryLoader::serialize() const {
  new core::FatalError(core::ATH_NOT_IMPLEMENTED, "Not serializable");
  return ""; // suppress warning
}

} // namespace athena::loaders

namespace athena::core {
template <>
std::string
core::AbstractLoader::getLoaderName<athena::loaders::MemoryLoader>() {
  return "MemoryLoader";
}
} // namespace athena::core

extern "C" {
void MemoryLoaderLoad(void* loader, void* allocator, void* tensor) {
  auto pLoader = reinterpret_cast<athena::loaders::MemoryLoader*>(loader);
  auto pAllocator = reinterpret_cast<athena::core::Allocator*>(allocator);
  auto pTensor = reinterpret_cast<athena::core::inner::Tensor*>(tensor);

  athena::athena_assert(pLoader != nullptr, "Corrupted arg");
  athena::athena_assert(pAllocator != nullptr, "Corrupted arg");
  athena::athena_assert(pTensor != nullptr, "Corrupted arg");

  pLoader->load(pAllocator, pTensor);
}

void* CreateMemoryLoader(void* data, size_t size) {
  return new athena::loaders::MemoryLoader(data, size);
}
}