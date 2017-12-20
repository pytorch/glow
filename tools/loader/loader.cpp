// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "loader.h"
#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/IR/IR.h"
#include "glow/Importer/Caffe2.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

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
  auto imageH = localCopy.getHandle<>();

  auto dims = localCopy.dims();

  result->reset(ElemKind::FloatTy, {1, 3, dims[0], dims[1]});
  auto RH = result->getHandle<>();

  if (!opts::NoBGR) {
    // Convert to BGR.
    for (unsigned z = 0; z < 3; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          RH.at({0, 2 - z, x, y}) = (imageH.at({x, y, z}));
        }
      }
    }
  }
}

namespace opts {

llvm::cl::list<std::string>
    InputImageFilenames(llvm::cl::Positional,
                        llvm::cl::desc("<input image files>"),
                        llvm::cl::OneOrMore);

llvm::cl::OptionCategory ModelInputCat("How to input the models",
                                       "These control the caffe2 model paths");
llvm::cl::opt<std::string>
    NetDescFilename("network",
                    llvm::cl::desc("Specify the network structure file"),
                    llvm::cl::value_desc("netDescFilename"),
                    llvm::cl::cat(ModelInputCat), llvm::cl::Optional);
llvm::cl::alias NetDescFileNameA("n", llvm::cl::desc("Alias for -network"),
                                 llvm::cl::aliasopt(NetDescFilename),
                                 llvm::cl::cat(ModelInputCat));
llvm::cl::opt<std::string>
    NetWeightFilename("weight",
                      llvm::cl::desc("Specify the network weight file"),
                      llvm::cl::value_desc("netWeightFilename"),
                      llvm::cl::cat(ModelInputCat), llvm::cl::Optional);
llvm::cl::alias NetWeightFileNameA("w", llvm::cl::desc("Alias for -weight"),
                                   llvm::cl::aliasopt(NetWeightFilename),
                                   llvm::cl::cat(ModelInputCat));
llvm::cl::opt<std::string> NetDirectory(
    "directory",
    llvm::cl::desc("Specify the directory with the network structure "
                   "<init_net.pb> and weight <predict_net.pb> files"),
    llvm::cl::value_desc("netDirectory"), llvm::cl::cat(ModelInputCat),
    llvm::cl::Optional);
llvm::cl::alias NetDirectoryA("d", llvm::cl::desc("Alias for -directory"),
                              llvm::cl::aliasopt(NetDirectory),
                              llvm::cl::cat(ModelInputCat));

llvm::cl::OptionCategory
    ModelExportCat("How to export the Glow Intermediate Representation/Graphs",
                   "These options are for debugging the "
                   "graphs by writing the IR/Graphs to "
                   "given files/stdout");

llvm::cl::opt<std::string> DumpGraphDAGFile(
    "dumpGraphDAG",
    llvm::cl::desc("Specify the file to export the Graph in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(ModelExportCat));

llvm::cl::opt<bool> DumpGraph("dumpGraph",
                              llvm::cl::desc("Prints Graph to stdout"),
                              llvm::cl::cat(ModelExportCat));

llvm::cl::opt<std::string> DumpIRDAGFile(
    "dumpIRDAG",
    llvm::cl::desc("Specify the file to export the IR in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(ModelExportCat));

llvm::cl::opt<bool> DumpIR("dumpIR", llvm::cl::desc("Prints IR to stdout"),
                           llvm::cl::cat(ModelExportCat));

llvm::cl::opt<ImageNormalizationMode> ImageMode(
    "image_mode", llvm::cl::desc("Specify the image mode:"), llvm::cl::Required,
    llvm::cl::values(clEnumValN(ImageNormalizationMode::k0to1, "0to1",
                                "Values are in the range: 0 and 1"),
                     clEnumValN(ImageNormalizationMode::k0to256, "0to256",
                                "Values are in the range: 0 and 256"),
                     clEnumValN(ImageNormalizationMode::k128to127, "128to127",
                                "Values are in the range: -128 .. 127")));
llvm::cl::alias ImageModeA("i", llvm::cl::desc("Alias for -image_mode"),
                           llvm::cl::aliasopt(ImageMode));

llvm::cl::opt<bool>
    Verbose("verbose",
            llvm::cl::desc("Specify whether to run with verbose output"),
            llvm::cl::Optional);

llvm::cl::opt<bool>
    Timer("timer",
          llvm::cl::desc("Print timer output to stderr detailing how long it "
                         "takes for the program to execute"),
          llvm::cl::Optional);

llvm::cl::opt<bool>
    NoBGR("noBGR",
          llvm::cl::desc("Do not convert the given png files to BGR format"),
          llvm::cl::init(false));
} // namespace opts

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");

  Tensor data;
  Tensor expected_softmax(ElemKind::IndexTy, {1, 1});

  for (const auto &InputImageFilename : opts::InputImageFilenames) {
    if (opts::Verbose) {
      llvm::outs() << "loading and preprocessing: " + InputImageFilename +
                          "...\n";
    }
    loadImageAndPreprocess(InputImageFilename, &data, opts::ImageMode);

    if (!opts::NetDirectory.empty()) {
      opts::NetDescFilename.setValue(opts::NetDirectory + "/predict_net.pb");
      opts::NetWeightFilename.setValue(opts::NetDirectory + "/init_net.pb");
    }

    ExecutionEngine EE(BackendKind::Interpreter);
    SaveNode *SM;
    Variable *i0;
    Variable *i1;
    {
      caffe2ModelLoader LD(opts::NetDescFilename, opts::NetWeightFilename,
                           {"data", "gpu_0/data", "softmax_expected"},
                           {&data, &data, &expected_softmax}, EE);
      SM = LD.getRoot();
      i0 = llvm::cast<Variable>(LD.getOrCreateNodeByName("gpu_0/data"));
      i1 = llvm::cast<Variable>(LD.getOrCreateNodeByName("data"));
    }

    auto &G = EE.getGraph();
    auto &M = EE.getModule();

    if (opts::DumpGraph) {
      G.dump();
    }
    if (!opts::DumpGraphDAGFile.empty()) {
      G.dumpDAG(opts::DumpGraphDAGFile.c_str());
    }
    if (opts::DumpIR) {
      M.dump();
    }
    if (!opts::DumpIRDAGFile.empty()) {
      M.dumpDAG(opts::DumpIRDAGFile.c_str());
    }

    llvm::Timer timer("Infer", "Infer");
    if (opts::Timer)
      timer.startTimer();
    EE.run({i0, i1}, {&data, &data});
    if (opts::Timer)
      timer.stopTimer();

    Tensor &res = SM->getVariable()->getPayload();
    auto H = res.getHandle<>();
    Tensor slice = H.extractSlice(0);
    auto SH = slice.getHandle<>();

    llvm::outs() << "\n";

    llvm::outs() << "Model: " << opts::NetDescFilename << "\n"
                 << " File: " << InputImageFilename << " Result:" << SH.maxArg()
                 << "\n";
  }
  return 0;
}
