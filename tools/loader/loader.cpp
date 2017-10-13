// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Importer/Caffe2.h"

#include <iostream>

using namespace glow;

enum class ImageNormalizationMode {
  k0to1,     // Values are in the range: 0 and 1.
  k0to256,   // Values are in the range: 0 and 256.
  k128to127, // Values are in the range: -128 .. 127
};

ImageNormalizationMode strToImageNormalizationMode(const std::string &str) {
  if (str == "0to1")
    return ImageNormalizationMode::k0to1;
  if (str == "0to256")
    return ImageNormalizationMode::k0to256;
  if (str == "128to127")
    return ImageNormalizationMode::k128to127;

  GLOW_ASSERT(false && "Unknown image format");
}

/// Convert the normalization to numeric floating poing ranges.
std::pair<float, float> normModeToRange(ImageNormalizationMode mode) {
  switch (mode) {
  case ImageNormalizationMode::k0to1:
    return {0., 1.0};
  case ImageNormalizationMode::k0to256:
    return {0., 256.0};
  case ImageNormalizationMode::k128to127:
    return {-128., 128.};
  }

  GLOW_ASSERT(false && "Unknown image format");
}

/// Loads and normalizes a PNG into a tensor in the NCHW 3x224x224 format.
void loadImageAndPreprocess(const std::string &filename, Tensor *result,
                            ImageNormalizationMode normMode) {
  auto range = normModeToRange(normMode);

  Tensor localCopy;
  readPngImage(&localCopy, filename.c_str(), range);
  auto imageH = localCopy.getHandle<FloatTy>();

  auto dims = localCopy.dims();

  result->reset(ElemKind::FloatTy, {1, 3, dims[0], dims[1]});
  auto RH = result->getHandle<FloatTy>();

  // Convert to BGR.
  for (unsigned z = 0; z < 3; z++) {
    for (unsigned y = 0; y < dims[1]; y++) {
      for (unsigned x = 0; x < dims[0]; x++) {
        RH.at({0, 2 - z, x, y}) = (imageH.at({x, y, z}));
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " image.png [0to1 / 0to256 / 128to127]"
              << " network_structure.pb weights.pb\n";
    return -1;
  }

  Tensor data;
  Tensor expected_softmax(ElemKind::IndexTy, {1, 1});

  auto imageMode = strToImageNormalizationMode(argv[2]);
  loadImageAndPreprocess(argv[1], &data, imageMode);

  ExecutionEngine EE;
  caffe2ModelLoader LD(argv[3], argv[4],
                       {"data", "gpu_0/data", "softmax_expected"},
                       {&data, &data, &expected_softmax}, EE);

  EE.initVars();

  auto *SM = LD.getRoot();
  auto *i0 = cast<Variable>(LD.getOrCreateNodeByName("gpu_0/data"));
  auto *i1 = cast<Variable>(LD.getOrCreateNodeByName("data"));

  EE.infer({i0, i1}, {&data, &data});
  auto *res = EE.getTensor(SM);
  auto H = res->getHandle<FloatTy>();
  H.dump("res = ", "\n");
  Tensor slice = H.extractSlice(0);
  auto SH = slice.getHandle<FloatTy>();

  std::cout << "\n";
  std::cout << "Result = " << SH.maxArg() << "\n";

  return 0;
}
