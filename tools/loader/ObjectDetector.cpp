/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#include "Loader.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "ExecutorCore.h"
#include "ExecutorCoreHelperFunctions.h"

using namespace glow;

/// Dumps txt files for each output and for each file.
static int
processAndPrintResults(const llvm::StringMap<Placeholder *> &PHM,
                       PlaceholderBindings &bindings,
                       llvm::ArrayRef<std::string> inputImageBatchFilenames) {
  // Print out the object detection results.
  std::vector<Tensor *> vecOutTensors;
  std::vector<std::string> VecOutNames;
  for (auto const &OutEntry : PHM) {
    VecOutNames.push_back(OutEntry.getKey().str());
    Placeholder *SMPH = OutEntry.getValue();
    Tensor *SMT = bindings.get(SMPH);
    vecOutTensors.push_back(SMT);
  }
  for (unsigned i = 0; i < inputImageBatchFilenames.size(); i++) {
    llvm::outs() << "Input File " << inputImageBatchFilenames[i] << ":\n";
    for (size_t k = 0; k < VecOutNames.size(); k++) {
      std::error_code EC;
      int idx = inputImageBatchFilenames[i].find_last_of("\\/");
      int idx1 = inputImageBatchFilenames[i].find_last_of(".");
      std::string basename =
          inputImageBatchFilenames[i].substr(idx + 1, idx1 - idx - 1);

      // Dump all the output tensors of input file with name <inimage> to
      // files with name <inimage>_<tensorname>.txt in the working
      // directory.
      std::replace_if(VecOutNames[k].begin(), VecOutNames[k].end(),
                      [](char c) { return !isalnum(c); }, '_');
      std::string filename = basename + "_" + VecOutNames[k] + ".txt";
      llvm::raw_fd_ostream fd(filename, EC);
      if (EC) {
        llvm::outs() << "Error opening file " << filename;
        llvm::outs().flush();
        fd.close();
        return 1;
      }

      auto getDumpTensor = [](Tensor *t, int slice) {
        assert(t->dims().size() > 1 && "Tensor dims should be > 2");
        switch (t->getElementType()) {
        case ElemKind::Int64ITy:
          return t->getHandle<int64_t>().extractSlice(slice);
        case ElemKind::Int32ITy:
          return t->getHandle<int32_t>().extractSlice(slice);
        default:
          return t->getHandle<float>().extractSlice(slice);
        }
      };
      Tensor tensor = getDumpTensor(vecOutTensors[k], i);
      tensor.dump(fd, tensor.size());

      llvm::outs() << "\t" << filename << ":" << tensor.size() << "\n";
      fd.close();
    }
  }
  return 0;
}

/// Given the output PlaceHolder StringMap \p PHM, outputs results in to text
/// files for each output.
class ObjectDetectionProcessResult : public PostProcessOutputDataExtension {
public:
  int processOutputs(const llvm::StringMap<Placeholder *> &PHM,
                     PlaceholderBindings &bindings,
                     VecVecRef<std::string> imageList) override {
    processAndPrintResults(PHM, bindings, imageList[0]);
    return 0;
  }
};

int main(int argc, char **argv) {
  glow::Executor core("ObjectDetector", argc, argv);

  auto printResultCreator =
      []() -> std::unique_ptr<PostProcessOutputDataExtension> {
    return std::make_unique<ObjectDetectionProcessResult>();
  };
  core.registerPostProcessOutputExtension(printResultCreator);

  int numErrors = core.executeNetwork();
  return numErrors;
}
