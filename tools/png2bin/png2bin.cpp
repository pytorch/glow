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

#include "glow/Base/IO.h"
#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/Support/CommandLine.h"

#include <cmath>

using namespace glow;

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::Required);
llvm::cl::opt<std::string> OutputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<output file>"),
                                          llvm::cl::Required);
llvm::cl::OptionCategory QuantizationCat("Quantization Options");
llvm::cl::opt<float> QuantizationScale(
    "scale",
    llvm::cl::desc("Quantization scale. If NaN, no quantization is performed."),
    llvm::cl::init(NAN), llvm::cl::cat(QuantizationCat));
llvm::cl::opt<int> QuantizationOffset("offset",
                                      llvm::cl::desc("Quantization offset"),
                                      llvm::cl::init(0),
                                      llvm::cl::cat(QuantizationCat));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  const char *filename = argv[1];
  const char *outfile = argv[2];
  llvm::ArrayRef<float> mean = zeroMean;
  llvm::ArrayRef<float> std = oneStd;

  if (useImagenetNormalization) {
    mean = imagenetNormMean;
    std = imagenetNormStd;
  }

  Tensor png = readPngImageAndPreprocess(
      filename, imageNormMode, imageChannelOrder, imageLayout, mean, std);
  if (!std::isnan(static_cast<float>(QuantizationScale))) {
    TensorQuantizationParams TQP{QuantizationScale, QuantizationOffset};
    png = quantization::quantizeTensor(png, TQP);
  }
  writeToFile(png, outfile);
  return 0;
}
