// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/IR/IR.h"
#include "glow/Importer/Caffe2.h"
#include "glow/Quantization/Serialization.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

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
    return {-128., 127.};
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

  // Get first image's dimensions and check if grayscale or color.
  size_t imgHeight, imgWidth;
  bool isGray;
  std::tie(imgHeight, imgWidth, isGray) = getPngInfo(filenames[0].c_str());
  const size_t numChannels = isGray ? 1 : 3;

  // N x C x H x W
  result->reset(ElemKind::FloatTy,
                {numImages, numChannels, imgHeight, imgWidth});
  auto RH = result->getHandle<>();
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < filenames.size(); n++) {
    Tensor localCopy;
    bool loadSuccess = !readPngImage(&localCopy, filenames[n].c_str(), range);
    GLOW_ASSERT(loadSuccess && "Error reading input image.");
    auto imageH = localCopy.getHandle<>();

    auto dims = localCopy.dims();
    assert((dims[0] == imgHeight && dims[1] == imgWidth) &&
           "All images must have the same Height and Width");
    assert(dims[2] == numChannels &&
           "All images must have the same number of channels");

    // Convert to BGR, as this is what Caffe2 is expecting.
    for (unsigned z = 0; z < numChannels; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          RH.at({n, numChannels - 1 - z, x, y}) = (imageH.at({x, y, z}));
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

llvm::cl::opt<std::string> QuantizationProfileFile(
    "profile",
    llvm::cl::desc("Perform quantization profiling for a given graph "
                   "and save result to the file."),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional);

llvm::cl::opt<std::string> LoadProfileFile(
    "load_profile",
    llvm::cl::desc("Load quantization profile file and quantize the graph"),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional);

llvm::cl::opt<BackendKind> ExecutionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter"),
                     clEnumValN(BackendKind::JIT, "jit", "Use JIT"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter));

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");

  Tensor data;
  Tensor expectedSoftmax(ElemKind::IndexTy, {1, 1});

  loadImagesAndPreprocess(InputImageFilenames, &data, ImageMode);

  if (!NetDirectory.empty()) {
    NetDescFilename.setValue(NetDirectory + "/predict_net.pb");
    NetWeightFilename.setValue(NetDirectory + "/init_net.pb");
  }

  ExecutionEngine EE(ExecutionBackend);
  auto &G = *EE.getModule().createFunction(NetDirectory);
  SaveNode *SM;
  Variable *i0;
  Variable *i1;
  {
    caffe2ModelLoader LD(NetDescFilename, NetWeightFilename,
                         {"data", "gpu_0/data", "softmax_expected"},
                         {&data, &data, &expectedSoftmax}, G);
    SM = LD.getRoot();
    i0 = llvm::cast<Variable>(LD.getOrCreateNodeByName("gpu_0/data"));
    i1 = llvm::cast<Variable>(LD.getOrCreateNodeByName("data"));
  }

  assert(i0->getVisibilityKind() == Variable::VisibilityKind::Public);
  assert(i1->getVisibilityKind() == Variable::VisibilityKind::Public);

  // Instrument the graph to capture profiles for nodes' outputs.
  if (!QuantizationProfileFile.empty()) {
    ::profileQuantization(G);
  }

  // Quantize the graph based on the captured profile.
  if (!LoadProfileFile.empty()) {
    auto quantizationInfos = deserializeFromYaml(LoadProfileFile);
    ::generateQuantizedGraph(G, quantizationInfos);
  }

  // Emit IR for the graph.
  EE.compile(CompilationMode::Infer, &G);

  if (DumpGraph) {
    G.dump();
  }
  if (!DumpGraphDAGFile.empty()) {
    G.dumpDAG(DumpGraphDAGFile.c_str());
  }
  if (DumpIR) {
    EE.getIR().dump();
  }
  if (!DumpIRDAGFile.empty()) {
    EE.getIR().dumpDAG(DumpIRDAGFile.c_str());
  }

  llvm::Timer timer("Infer", "Infer");
  if (Timer)
    timer.startTimer();
  EE.run({i0, i1}, {&data, &data});
  if (Timer)
    timer.stopTimer();

  if (!QuantizationProfileFile.empty()) {
    std::vector<NodeQuantizationInfo> QI = generateNodeQuantizationInfos(G);
    serializeToYaml(QuantizationProfileFile, QI);
  }

  Tensor &res = SM->getVariable()->getPayload();
  auto H = res.getHandle<>();
  llvm::outs() << "Model: " << NetDescFilename << "\n";
  for (unsigned i = 0; i < InputImageFilenames.size(); i++) {
    Tensor slice = H.extractSlice(i);
    auto SH = slice.getHandle<>();
    llvm::outs() << " File: " << InputImageFilenames[i]
                 << " Result:" << SH.minMaxArg().second << "\n";
  }
  return 0;
}
