/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

using namespace glow;

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: png2bits INFILE OUTFILE\n");
    exit(1);
  }
  const char *filename = argv[1];
  const char *outfile = argv[2];
  Tensor png = readPngImageAndPreprocess(
      filename, ImageNormalizationMode::k0to1, ImageChannelOrder::BGR,
      ImageLayout::NCHW, false);
  TensorQuantizationParams TQP{1.0f / 127.0f, 0};
  Tensor qpng = quantization::quantizeTensor(png, TQP);
  writeToFile(qpng, outfile);
  return 0;
}
