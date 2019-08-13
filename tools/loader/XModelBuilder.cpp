/** Copyright 2019 Xperi Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License”); 
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Loader.h"

#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <sstream>

llvm::cl::OptionCategory inputLoaderCat("Input Loader Options");

llvm::cl::opt<std::string> modelInputName(
    "model-input-name",
    llvm::cl::desc("The name of the variable for the model's input data."),
    llvm::cl::value_desc("string"), llvm::cl::Required,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::list<std::string> inputFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<input file(s)> Input file name(s) from which input is read. "
                   "Input is read byte-wise, so the file is assumed to be "
                   "a byte-stream. For instance, if the input tensor is a "
                   "2x3 matrix of 32-bit floats, then the file is expected "
                   "to contain 4x2x3 = 48 bytes. The values are loaded into "
                   "the tensor column-wise: (1, 1), (1, 2), (1, 3), (2, 1), ..., (3, 3)"),
    llvm::cl::value_desc("space-separated strings"),
    llvm::cl::ZeroOrMore);

llvm::cl::list<unsigned> inputTensorDimensions(
    "input-tensor-dims",
    llvm::cl::desc("Comma-separated list of input tensor dimensions"),
    llvm::cl::value_desc("unsigned int"),
    llvm::cl::OneOrMore,
    llvm::cl::CommaSeparated,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::list<std::string> outputTensorNames(
    "output-tensor-names",
    llvm::cl::desc("Comma-separated list of output tensor names"),
    llvm::cl::value_desc("list of strings"),
    llvm::cl::OneOrMore,
    llvm::cl::CommaSeparated,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<std::string> inputFileList(
    "input-file-list",
    llvm::cl::desc(
        "Name of the file containing list of files (one per line) to process. This "
        "is equivalent to passing each file name individually. "),
    llvm::cl::value_desc("string"),
    llvm::cl::Optional,
    llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-input-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(inputLoaderCat));

llvm::cl::opt<bool> writeOutput(
    "write-output",
    llvm::cl::desc(
        "Write output of the inference (only applicable when not building a bundle."),
    llvm::cl::cat(inputLoaderCat));

static std::unique_ptr<glow::ProtobufLoader> 
createProtobufLoader(glow::Loader &loader, const glow::TypeRef inputType)
{
    std::unique_ptr<glow::ProtobufLoader> ptbLoader{};
    const bool caffe2Model{!loader.getCaffe2NetDescFilename().empty()};

    if (caffe2Model)
        ptbLoader.reset(new glow::Caffe2ModelLoader(
            loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
            {modelInputName.c_str()}, {inputType}, *loader.getFunction()));
    else
        ptbLoader.reset(new glow::ONNXModelLoader(loader.getOnnxModelFilename(), {modelInputName.c_str()},
            {inputType}, *loader.getFunction()));

    return ptbLoader;
}

static std::pair<glow::Placeholder *, std::unordered_map<std::string, glow::Tensor *>>
buildNetwork(glow::Loader &loader, const glow::TypeRef inputType, glow::PlaceholderBindings &ioBindings)
{
    std::unique_ptr<glow::ProtobufLoader> LD{};
    const char *inputName{modelInputName.c_str()};
    glow::Placeholder *inputPH{};
    glow::Placeholder *outputPH{};
    glow::Tensor *outputTensor{};
    std::pair<glow::Placeholder *, std::unordered_map<std::string, glow::Tensor *>> ret{};

    LD = createProtobufLoader(loader, inputType);
    (void) ioBindings.allocate(loader.getModule()->getPlaceholders());

    if (convertInAndOutToFp16) {
        glow::PrecisionConfiguration precConfig;
        glow::TypeAToTypeBFunctionConverter converter(
            *loader.getFunction(), glow::ElemKind::FloatTy,
            glow::ElemKind::Float16Ty, precConfig);
        for (auto *placeholder : loader.getModule()->getPlaceholders())
            converter.convertPlaceholder(*placeholder, &ioBindings);
    }

    loader.compile(ioBindings);

    inputPH = llvm::cast<glow::Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName(inputName)));
    ret.first = inputPH;

    for (const std::string &name : outputTensorNames) {
        outputPH = EXIT_ON_ERR(LD->getOutputByName(name));
        outputTensor = ioBindings.get(outputPH);
        ret.second.insert(std::make_pair(name, outputTensor));
    }

    return ret;
}

static void
gatherFiles(std::vector<std::string> &files)
{
    for (auto file : inputFilenames)
        files.push_back(file);
    if (inputFileList.size() != 0) {
        std::ifstream fstrm{inputFileList};
        if (!fstrm) {
            llvm::errs() << "Error processing input file list " << inputFileList << "\n";
            exit(1);
        }

        std::string file{};
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

static void
loadInputData(const std::string &file, std::vector<char> &inputData, std::size_t size)
{
    std::ifstream inputFile(file.c_str(), std::ios::binary);
    inputFile.seekg(0, std::ios::end);

    if (inputFile.tellg() != size) {
        llvm::errs() << "Size of " << file << " does not match expected size " << size << "\n";
        exit(1);
    }

    inputFile.seekg(0, std::ios::beg);
    inputFile.read(inputData.data(), size);
}

static void
runInference(glow::Loader &loader,
             glow::PlaceholderBindings &ioBindings, 
             std::pair<glow::Placeholder *, std::unordered_map<std::string, glow::Tensor *>> &ioPlaceholders,
             const std::vector<char> &inputData,
             std::unordered_map<std::string, std::pair<std::vector<char>, std::size_t>> &outputData)
{
    glow::Tensor *inputT = ioBindings.get(ioPlaceholders.first);

    std::memcpy(inputT->getUnsafePtr(), inputData.data(), inputData.size());

    if (convertInAndOutToFp16)
        inputT->convertToType(glow::ElemKind::Float16Ty);

    loader.runInference(ioBindings, 1);

    for (auto &keyval : ioPlaceholders.second) {
        outputData.insert(std::make_pair(keyval.first, std::make_pair(std::vector<char>{}, 0)));
        outputData[keyval.first].first.reserve(ioPlaceholders.second[keyval.first]->getSizeInBytes());
        outputData[keyval.first].second = ioPlaceholders.second[keyval.first]->getSizeInBytes();
        std::memcpy(outputData[keyval.first].first.data(), ioPlaceholders.second[keyval.first]->getUnsafePtr(), 
                    ioPlaceholders.second[keyval.first]->getSizeInBytes());
    }
}

static void
writeOutputData(const std::unordered_map<std::string, std::pair<std::vector<char>, std::size_t>> &outputData, 
                const std::string &file)
{
    if (writeOutput) {
        std::ofstream outputFile{};
        std::string name{};

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

int 
main(int argc, char **argv)
{
    glow::PlaceholderBindings ioBindings{};                        // IO Bindings
    std::pair<glow::Placeholder *,
              std::unordered_map<std::string, glow::Tensor *>>
        ioPlaceholders{};                                         // first = input placeholder,
                                                                  // second = <output name, output tensor>

    // This must be called before a loader instance is created.
    glow::parseCommandLine(argc, argv);
    glow::Loader loader{};

    std::vector<std::size_t> dims{};
    std::vector<char> inputData{};
    std::vector<std::string> files{};
    glow::Tensor inputT{};

    for (auto dim : inputTensorDimensions)
        dims.push_back(dim);

    inputT.reset(glow::ElemKind::FloatTy, dims);
    ioPlaceholders = buildNetwork(loader, &inputT.getType(), ioBindings);

    if (glow::emittingBundle()) {
        if (!inputFileList.empty() || inputFilenames.size() != 0)
            llvm::errs() << "WARNING: input files specification has no effect when emitting bundle.\n";
        return 0;
    }

    if (inputFileList.empty() && inputFilenames.size() == 0) {
        llvm::errs()
            << "Args: Either positional <input file(s)> or -input-file-list "
            "must be used to specify input data when not outputting bundle.\n";
        std::exit(1);
    }

    if (!inputFileList.empty() && inputFilenames.size() != 0) {
        llvm::errs()
            << "Args: Either positional <input file(s)> or -input-file-list (but not both) "
            "must be used to specify input data.\n";
        std::exit(1);
    }

    // Stores the list of files containing input in "files".
    gatherFiles(files);
    for (auto &file : files) {
        inputData.clear();
        // The size of input is computed from input dimensions, known from command line
        // arguments, and the size of float.
        inputData.reserve(inputT.getSizeInBytes());
        // Every output is identified by its name (std::string), and is stored in 
        // a byte array; it also carries information about its size. So
        // first = name
        // second = <byte array, array size>.
        std::unordered_map<std::string, std::pair<std::vector<char>, std::size_t>> outputData{};

        // Reads input from file to the inputData vector, of max size = the capacity of
        // the input tensor.
        loadInputData(file, inputData, inputT.getSizeInBytes());
        // Output data is stored in outputData.
        runInference(loader, ioBindings, ioPlaceholders, inputData, outputData);
        // Writes output to a file whose base name is given by "file".
        writeOutputData(outputData, file);
    }

    // Are we profiling? If so, spit out the profile.
    if (glow::profilingGraph())
        loader.generateAndSerializeQuantizationInfos(ioBindings);

    return 0;
}
