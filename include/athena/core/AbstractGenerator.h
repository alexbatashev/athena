/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
#ifndef ATHENA_ABSTRACTGENERATOR_H
#define ATHENA_ABSTRACTGENERATOR_H

#include <athena/core/Tensor.h>

namespace athena::core {

class AbstractGenerator {
    public:
    virtual void generateAllocation(Tensor &a)                = 0;
    virtual void generateAdd(Tensor &a, Tensor &b, Tensor &c) = 0;
};

}  // namespace athena::core

#endif  // ATHENA_ABSTRACTGENERATOR_H
