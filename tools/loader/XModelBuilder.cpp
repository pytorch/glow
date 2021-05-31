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

/**
 * Contributed by Xperi Corporation on August 13, 2019
 */

#include "Loader.h"

#include "glow/Base/Tensor.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>

using namespace glow;

namespace {
llvm::cl::OptionCategory inputLoaderCat("Input Loader Options");

llvm::cl::opt<std::string> modelInputName(
    "model-input-name",
    llvm::cl::desc("The name of the variable for the model's input data."),
    llvm::cl::value_desc("string"), llvm::cl::Required,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::list<std::string> inputFilenames(
    llvm::cl::Positional,
    llvm::cl::desc(
        "<input file(s)> Input file name(s) from which input is read. "
        "Input is read byte-wise, so the file is assumed to be "
        "a byte-stream. For instance, if the input tensor is a "
        "2x3 matrix of 32-bit floats, then the file is expected "
        "to contain 4x2x3 = 48 bytes. The values are loaded into "
        "the tensor column-wise: (1, 1), (1, 2), (1, 3), (2, 1), ..., (3, 3)"),
    llvm::cl::value_desc("space-separated strings"), llvm::cl::ZeroOrMore);

llvm::cl::list<unsigned> inputTensorDimensions(
    "input-tensor-dims",
    llvm::cl::desc("Comma-separated list of input tensor dimensions"),
    llvm::cl::value_desc("unsigned int"), llvm::cl::OneOrMore,
    llvm::cl::CommaSeparated, llvm::cl::cat(inputLoaderCat));

llvm::cl::list<std::string> outputTensorNames(
    "output-tensor-names",
    llvm::cl::desc("Comma-separated list of output tensor names"),
    llvm::cl::value_desc("list of strings"), llvm::cl::OneOrMore,
    llvm::cl::CommaSeparated, llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<std::string> inputFileList(
    "input-file-list",
    llvm::cl::desc("Name of the file containing list of files (one per line) "
                   "to process. This "
                   "is equivalent to passing each file name individually. "),
    llvm::cl::value_desc("string"), llvm::cl::Optional,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-input-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<bool>
    writeOutput("write-output",
                llvm::cl::desc("Write output of the inference (only applicable "
                               "when not building a bundle."),
                llvm::cl::cat(inputLoaderCat));
} // unnamed namespace

/// Creates and \returns the ProtobufLoader given \p loader and the
/// \p inputType.
static std::unique_ptr<ProtobufLoader>
createProtobufLoader(Loader &loader, const TypeRef inputType) {
  std::unique_ptr<ProtobufLoader> ptbLoader;
  const bool caffe2Model{!loader.getCaffe2NetDescFilename().empty()};

  if (caffe2Model) {
    ptbLoader.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename().str(),
        loader.getCaffe2NetWeightFilename().str(), {modelInputName.c_str()},
        {inputType}, *loader.getFunction()));
  } else {
    ptbLoader.reset(new ONNXModelLoader(loader.getOnnxModelFilename().str(),
                                        {modelInputName.c_str()}, {inputType},
                                        *loader.getFunction()));
  }

  return ptbLoader;
}

/// Builds the network and \returns a pair of the form (input placeholder ptr,
/// (output name, output tensor ptr)). \p loader - the loader \p inputType - the
/// input type \p ioBindings - a reference to the placeholder bindings
/// (allocated in this function)
static std::pair<Placeholder *, std::unordered_map<std::string, Tensor *>>
buildNetwork(Loader &loader, const TypeRef inputType,
             PlaceholderBindings &ioBindings) {
  std::unique_ptr<ProtobufLoader> LD;
  const char *inputName{modelInputName.c_str()};
  Placeholder *inputPH;
  Placeholder *outputPH;
  Tensor *outputTensor;
  std::pair<Placeholder *, std::unordered_map<std::string, Tensor *>> ret;

  // Create the protobuf loader and allocate io bindings.
  LD = createProtobufLoader(loader, inputType);
  (void)ioBindings.allocate(loader.getModule()->getPlaceholders());

  // Convert to Fp16 if required.
  if (convertInAndOutToFp16) {
    PrecisionConfiguration precConfig;
    TypeAToTypeBFunctionConverter converter(*loader.getFunction(),
                                            ElemKind::FloatTy,
                                            ElemKind::Float16Ty, precConfig);
    for (auto *placeholder : loader.getModule()->getPlaceholders())
      converter.convertPlaceholder(*placeholder, &ioBindings);
  }

  // Compile the network
  loader.compile(ioBindings);

  // Grab the input placeholder
  inputPH =
      llvm::cast<Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName(inputName)));
  ret.first = inputPH;

  // Grab all output placeholders by name/tensor
  for (const std::string &name : outputTensorNames) {
    outputPH = EXIT_ON_ERR(LD->getOutputByName(name));
    outputTensor = ioBindings.get(outputPH);
    ret.second.insert(std::make_pair(name, outputTensor));
  }

  return ret;
}

/// Gathers input from the files specified (either a single file containing one
/// input file name per line, or multiple input files) \p files - a reference of
/// type std::vector<std::string> that contains the gathered input filenames.
static void gatherFiles(std::vector<std::string> &files) {
  // Grab any files specified on the command line as positional arguments.
  for (auto file : inputFilenames) {
    files.push_back(file);
  }
  // If a file with input file names was specified, read the input file names
  // from the specified file.
  if (inputFileList.size() != 0) {
    std::ifstream fstrm{inputFileList};
    if (!fstrm) {
      llvm::errs() << "Error processing input file list " << inputFileList
                   << "\n";
      exit(1);
    }

    std::string file;
    while (std::getline(fstrm, file)) {
      files.push_back(file);
      std::ifstream check{file};
      if (!check) {
        llvm::errs() << "Error processing input file " << file << "\n";
        exit(1);
      }
    }
  }
}

/// Loads input data of size \p size from a given \p file into \p inputData.
static void loadInputData(const std::string &file, std::vector<char> &inputData,
                          std::size_t size) {
  std::ifstream inputFile(file.c_str(), std::ios::binary);
  inputFile.seekg(0, std::ios::end);

  if (inputFile.tellg() != long(size)) {
    llvm::errs() << "Size of " << file << " does not match expected size "
                 << size << "\n";
    exit(1);
  }

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(inputData.data(), size);
}

/// Run inference given the created \p loader, \p ioBindings, and \p
/// ioPlaceholders. \p inputData - the vector containing our input data (as raw
/// bytes) \p outputData - a pair of the form (output tensor name, (output
/// bytes, output size))
static void runInference(
    Loader &loader, PlaceholderBindings &ioBindings,
    std::pair<Placeholder *, std::unordered_map<std::string, Tensor *>>
        &ioPlaceholders,
    const std::vector<char> &inputData,
    std::unordered_map<std::string, std::pair<std::vector<char>, dim_t>>
        &outputData) {
  // Grab a pointer to the input tensor from the placeholders
  Tensor *inputT = ioBindings.get(ioPlaceholders.first);

  // Copy the raw input data from inputData into the input tensor.
  std::memcpy(inputT->getUnsafePtr(), inputData.data(), inputData.size());

  // If we must first convert to Fp16, do so.
  if (convertInAndOutToFp16) {
    inputT->convertToType(ElemKind::Float16Ty);
  }

  // Finally, run inference. The input data is already stored inside the input
  // tensor, inside the ioBindings. The batch size is 1.
  loader.runInference(ioBindings, 1);

  // Finally, store our output - we may have multiple output tensors, so sort
  // the output into the correct named output bins.
  for (auto &keyval : ioPlaceholders.second) {
    outputData.insert(
        std::make_pair(keyval.first, std::make_pair(std::vector<char>{}, 0)));
    outputData[keyval.first].first.reserve(
        ioPlaceholders.second[keyval.first]->getSizeInBytes());
    outputData[keyval.first].second =
        ioPlaceholders.second[keyval.first]->getSizeInBytes();
    std::memcpy(outputData[keyval.first].first.data(),
                ioPlaceholders.second[keyval.first]->getUnsafePtr(),
                ioPlaceholders.second[keyval.first]->getSizeInBytes());
  }
}

/// Write out \p outputData into \p file.
static void writeOutputData(
    const std::unordered_map<std::string, std::pair<std::vector<char>, dim_t>>
        &outputData,
    const std::string &file) {
  if (writeOutput) {
    std::ofstream outputFile;
    std::string name;

    // The output file is formated as [input file name].[output tensor
    // name].out.dat
    for (auto &keyval : outputData) {
      name = file;
      name += ".";
      name += keyval.first;
      name += ".out.dat";

      outputFile.open(name.c_str(), std::ios::out | std::ios::binary);
      if (!outputFile) {
        std::cerr << "Unable to open output file: " << name << std::endl;
        return;
      }
      outputFile.write(keyval.second.first.data(), keyval.second.second);
      outputFile.close();
    }
  }
}

int main(int argc, char **argv) {
  PlaceholderBindings ioBindings; // IO Bindings
  std::pair<Placeholder *, std::unordered_map<std::string, Tensor *>>
      ioPlaceholders; // first = input placeholder,
                      // second = <output name, output tensor>

  // This must be called before a loader instance is created.
  parseCommandLine(argc, argv);
  Loader loader;

  std::vector<dim_t> dims;
  std::vector<char> inputData;
  std::vector<std::string> files;
  Tensor inputT;

  for (auto dim : inputTensorDimensions) {
    dims.push_back(dim);
  }

  inputT.reset(ElemKind::FloatTy, dims);
  ioPlaceholders = buildNetwork(loader, &inputT.getType(), ioBindings);

  if (emittingBundle()) {
    if (!inputFileList.empty() || inputFilenames.size() != 0) {
      llvm::errs() << "WARNING: input files specification has no effect when "
                      "emitting bundle.\n";
    }
    return 0;
  }

  if (inputFileList.empty() && inputFilenames.size() == 0) {
    llvm::errs()
        << "Args: Either positional <input file(s)> or -input-file-list "
           "must be used to specify input data when not outputting bundle.\n";
    std::exit(1);
  }

  if (!inputFileList.empty() && inputFilenames.size() != 0) {
    llvm::errs() << "Args: Either positional <input file(s)> or "
                    "-input-file-list (but not both) "
                    "must be used to specify input data.\n";
    std::exit(1);
  }

  // Stores the list of files containing input in "files".
  gatherFiles(files);
  for (auto &file : files) {
    inputData.clear();
    // The size of input is computed from input dimensions, known from command
    // line arguments, and the size of float.
    inputData.reserve(inputT.getSizeInBytes());
    // Every output is identified by its name (std::string), and is stored in
    // a byte array; it also carries information about its size. So
    // first = name
    // second = <byte array, array size>.
    std::unordered_map<std::string, std::pair<std::vector<char>, dim_t>>
        outputData;

    // Reads input from file to the inputData vector, of max size = the capacity
    // of the input tensor.
    loadInputData(file, inputData, inputT.getSizeInBytes());
    // Output data is stored in outputData.
    runInference(loader, ioBindings, ioPlaceholders, inputData, outputData);
    // Writes output to a file whose base name is given by "file".
    writeOutputData(outputData, file);
  }

  // Are we profiling? If so, spit out the profile.
  if (profilingGraph()) {
    loader.generateAndSerializeProfilingInfos(ioBindings);
  }

  return 0;
}
