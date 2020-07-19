//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "CudaDevice.hpp"
#include "../utils/utils.hpp"
#include "CudaEvent.hpp"
#include "utils.hpp"

#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/LaunchCommand.h>

#include <array>
#include <fstream>
#include <nvvm.h>
#include <vector>

static auto compileNVVM(const std::vector<char>& nvvmIr) -> std::vector<char> {
  nvvmProgram compileUnit;
  nvvmResult res;

  nvvmCreateProgram(&compileUnit);
  nvvmAddModuleToProgram(compileUnit, nvvmIr.data(), nvvmIr.size(),
                         "LLVMDialectModule");

  const char* options[] = {"-arch=compute_35"};

  res = nvvmCompileProgram(compileUnit, 1, options);
  if (res != NVVM_SUCCESS) {
    // todo needs proper error handling
    std::cerr << res << '\n';
    std::cerr << nvvmIr.data() << '\n';
    size_t logSize;
    nvvmGetProgramLogSize(compileUnit, &logSize);
    char* msg = new char[logSize];
    nvvmGetProgramLog(compileUnit, msg);
    std::cerr << msg << "\n";
    delete[] msg;
    std::terminate();
  }

  size_t ptxSize = 0;
  nvvmGetCompiledResultSize(compileUnit, &ptxSize);

  std::vector<char> ptx;
  ptx.reserve(ptxSize);

  nvvmGetCompiledResult(compileUnit, ptx.data());

  nvvmDestroyProgram(&compileUnit);

  return ptx;
}

namespace polarai::backend::generic {
CudaDevice::CudaDevice(CUdevice device) : mDevice(device) {
  std::array<char, 100> name;
  check(cuDeviceGetName(name.data(), 100, mDevice));
  mDeviceName = std::string(name.data());

  check(cuCtxCreate(&mDeviceContext, 0, mDevice));
  check(cuStreamCreate(&mStream, 0));

  mAllocator = std::make_shared<CudaAllocator>();
}

std::string CudaDevice::getDeviceName() const { return mDeviceName; }

void CudaDevice::selectBinary(
    std::vector<std::shared_ptr<ProgramDesc>>& programs) {
  cuCtxSetCurrent(mDeviceContext);

  for (auto& module : programs) {
    if (module->type == ProgramDesc::Type::PTX) {
      mPtxModule = module;
      break;
    }
  }

  auto ptx = compileNVVM(mPtxModule->data);

  // todo real image selection logic
  std::array<char, 4096> jitErrorBuffer = {0};

  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* jitOptionsVals[] = {jitErrorBuffer.data(),
                            reinterpret_cast<void*>(jitErrorBuffer.size())};
  check(cuLinkCreate(2, jitOptions, jitOptionsVals, &linkState));
  check(cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                      static_cast<void*>(ptx.data()), ptx.size(), "kernels", 0,
                      nullptr, nullptr));
}
void CudaDevice::consumeEvent(Event* evt) {
  evt->wait();
  delete evt;
}

Event* CudaDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                          Event* blockingEvent) {
  cuCtxSetCurrent(mDeviceContext);
  uint64_t zero = 0;
  std::vector<void*> args;

  for (int i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor = static_cast<TensorInfo*>(cmd.args[i].arg);
      auto record = tensorInfoToRecord(tensor);
      auto buf = allocator.get<CUdeviceptr>(record, *this);
      args.push_back(buf);
      args.push_back(buf);

      args.push_back(&zero); // offset

      for (int dim = 0; dim < tensor->dims; dim++) {
        args.push_back(&tensor->shape[dim]); // size
      }
      for (int dim = 0; dim < tensor->dims; dim++) {
        args.push_back(&zero); // stride
      }
    } else {
      args.push_back(cmd.args[i].arg);
    }
  }

  size_t gridX = mPtxModule->kernels[cmd.kernelName].globalX;
  size_t gridY = mPtxModule->kernels[cmd.kernelName].globalY;
  size_t gridZ = mPtxModule->kernels[cmd.kernelName].globalZ;
  size_t blockX = mPtxModule->kernels[cmd.kernelName].localX;
  size_t blockY = mPtxModule->kernels[cmd.kernelName].localY;
  size_t blockZ = mPtxModule->kernels[cmd.kernelName].localZ;

  CUfunction func;
  check(cuModuleGetFunction(&func, mMainModule, cmd.kernelName));

  check(cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, 0,
                       mStream, args.data(), nullptr));
  check(cuStreamSynchronize(mStream));
  CUevent evt;
  check(cuEventCreate(&evt, 0));
  check(cuEventRecord(evt, mStream));

  return new CudaEvent(this, evt);
}
CudaDevice::~CudaDevice() {
  cuModuleUnload(mMainModule);
  cuCtxDestroy(mDeviceContext);
}
} // namespace polarai::backend::generic
