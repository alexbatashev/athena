#include <utility>

/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_OPERATION_H
#define ATHENA_OPERATION_H

#include <string>

namespace athena::core {
    class Operation {
     protected:
        std::string mName;
     public:
        explicit Operation(std::string&& name) : mName(std::move(name)) {};
        template <class Generator, typename ...Args>
        void gen(Generator g, Args... args) {};
        std::string getName();
    };
}

#endif //ATHENA_OPERATION_H
