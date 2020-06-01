
//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_MEMCPYLOADER_H
#define ATHENA_MEMCPYLOADER_H

#include <athena/core/loader/AbstractLoader.h>
#include <athena/loaders/internal/MemcpyLoaderInternal.h>

namespace athena::loaders {
namespace internal {
class MemcpyLoaderInternal;
}
class ATH_LOADERS_EXPORT MemcpyLoader {
public:
  using InternalType = internal::MemcpyLoaderInternal;
};
} // namespace athena::loaders

#endif // ATHENA_MEMCPYLOADER_H
