// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/IR/IR.h"
#include "glow/Importer/Caffe2.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define DEFAULT_CAFFE2_HEIGHT 224
#define DEFAULT_CAFFE2_WIDTH 224

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

/// Loads and normalizes all PNGs into a tensor in the NCHW 3x224x224 format.
void loadImagesAndPreprocess(const llvm::cl::list<std::string> &filenames,
                             Tensor *result, ImageNormalizationMode normMode) {
  assert(filenames.size() > 0 &&
         "There must be at least one filename in filenames");
  auto range = normModeToRange(normMode);
  unsigned numImages = filenames.size();
  // N x C x H x W
  result->reset(ElemKind::FloatTy,
                {numImages, 3, DEFAULT_CAFFE2_HEIGHT, DEFAULT_CAFFE2_WIDTH});
  auto RH = result->getHandle<>();
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < filenames.size(); n++) {
    Tensor localCopy;
    readPngImage(&localCopy, filenames[n].c_str(), range);
    auto imageH = localCopy.getHandle<>();

    auto dims = localCopy.dims();
    assert(
        (dims[0] == DEFAULT_CAFFE2_HEIGHT && dims[1] == DEFAULT_CAFFE2_WIDTH) &&
        "All images must have the same Height and Width");

    // Convert to BGR, as this is what Caffe2 is expecting.
    for (unsigned z = 0; z < 3; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          RH.at({n, 2 - z, x, y}) = (imageH.at({x, y, z}));
        }
      }
    }
  }
}

namespace {

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

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");

  Tensor data;
  Tensor expected_softmax(ElemKind::IndexTy, {1, 1});

  loadImagesAndPreprocess(InputImageFilenames, &data, ImageMode);

  if (!NetDirectory.empty()) {
    NetDescFilename.setValue(NetDirectory + "/predict_net.pb");
    NetWeightFilename.setValue(NetDirectory + "/init_net.pb");
  }

  ExecutionEngine EE(BackendKind::Interpreter);
  auto &G = EE.getGraph();
  auto &M = EE.getModule();
  SaveNode *SM;
  Variable *i0;
  Variable *i1;
  {
    caffe2ModelLoader LD(NetDescFilename, NetWeightFilename,
                         {"data", "gpu_0/data", "softmax_expected"},
                         {&data, &data, &expected_softmax}, G);
    SM = LD.getRoot();
    i0 = llvm::cast<Variable>(LD.getOrCreateNodeByName("gpu_0/data"));
    i1 = llvm::cast<Variable>(LD.getOrCreateNodeByName("data"));
  }

  // Hold a reference to i0 and i1 to prevent the optimizer from deleting them.
  // One of the nodes, 'gpu' or 'data' is unused and we don't want the optimizer
  // to delete the nodes because it makes it easier for us to initialize the
  // network.
  NodeValue ref0(i0,0);
  NodeValue ref1(i1,0);

  // Emit IR for the graph.
  EE.compile(CompilationMode::Infer);

  if (DumpGraph) {
    G.dump();
  }
  if (!DumpGraphDAGFile.empty()) {
    G.dumpDAG(DumpGraphDAGFile.c_str());
  }
  if (DumpIR) {
    M.dump();
  }
  if (!DumpIRDAGFile.empty()) {
    M.dumpDAG(DumpIRDAGFile.c_str());
  }

  llvm::Timer timer("Infer", "Infer");
  if (Timer)
    timer.startTimer();
  EE.run({i0, i1}, {&data, &data});
  if (Timer)
    timer.stopTimer();

  Tensor &res = SM->getVariable()->getPayload();
  auto H = res.getHandle<>();
  llvm::outs() << "Model: " << NetDescFilename << "\n";
  for (unsigned i = 0; i < InputImageFilenames.size(); i++) {
    Tensor slice = H.extractSlice(i);
    auto SH = slice.getHandle<>();
    llvm::outs() << " File: " << InputImageFilenames[i]
                 << " Result:" << SH.maxArg() << "\n";
  }
  return 0;
}
