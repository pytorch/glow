/**
 * Copyright (c) 2019-present, XPERI Corporation
 * 
 * Author: William Yessen <william.yessen@xperi.com>
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
    llvm::cl::desc("Comma-separated list of nput tensor dimensions"),
    llvm::cl::value_desc("unsigned int"),
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

llvm::cl::opt<std::string> logFile(
    "log-file",
    llvm::cl::desc(
        "Name of the file where output will be written (if not given, output is written to stdout). "
        "This option applies only if -write-output is supplied. "),
    llvm::cl::value_desc("string"),
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

static std::pair<glow::Placeholder *, glow::Tensor *> 
buildNetwork(glow::Loader &loader, const glow::TypeRef inputType, glow::PlaceholderBindings &ioBindings)
{
    std::unique_ptr<glow::ProtobufLoader> LD{};
    const char *inputName{modelInputName.c_str()};
    glow::Placeholder *inputPH{};
    glow::Placeholder *outputPH{};
    glow::Tensor *outputTensor{};
    std::pair<glow::Placeholder *, glow::Tensor *> ret{};

    LD = createProtobufLoader(loader, inputType);
    (void) ioBindings.allocate(loader.getModule()->getPlaceholders());

    if (convertInAndOutToFp16) {
        glow::TypeAToTypeBFunctionConverter converter(
            *loader.getFunction(), glow::ElemKind::FloatTy, glow::ElemKind::Float16Ty);
        for (auto *placeholder : loader.getModule()->getPlaceholders())
            converter.convertPlaceholder(*placeholder, &ioBindings);
    }

    loader.compile(ioBindings);
    inputPH = llvm::cast<glow::Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName(inputName)));
    outputPH = EXIT_ON_ERR(LD->getSingleOutput());
    outputTensor = ioBindings.get(outputPH);
    ret.first = inputPH;
    ret.second = outputTensor;

    return ret;
}

static std::size_t
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

        std::string line{};
        while (std::getline(fstrm, line))
            files.push_back(line);
    }

    std::size_t fileSize{};
    bool firstTime{true};
    for (auto file : files) {
        std::ifstream inputFile(file.c_str(), std::ios::binary|std::ios::ate);
        if (inputFile) {
            std::ifstream::pos_type size{inputFile.tellg()};
            if (firstTime) {
                firstTime = false;
                fileSize = size;
            } else {
                if (fileSize != size) {
                    llvm::errs() << "Some input filnes are of different sizes! Exiting.\n";
                    exit(1);
                }
            }
        } else {
            llvm::errs() << "Error processing input file: " << file << "\n";
            exit(1);
        }
    }

    return fileSize / sizeof(float_t);
}

static void
loadInputData(const std::vector<std::string> &files, std::size_t fileSize, std::vector<std::float_t> &inputData)
{
    inputData.push_back(0.0);
    inputData.push_back(0.0);

    inputData.reserve(fileSize * files.size());

    for (std::size_t jj = 0; jj < files.size(); ++jj) {
        std::ifstream inputFile(files[jj].c_str(), std::ios::binary);
        inputFile.read(reinterpret_cast<char *>(&(inputData[jj * fileSize])), fileSize * sizeof(std::float_t));
    }
}

static void
runInference(glow::Loader &loader,
             glow::PlaceholderBindings &ioBindings, 
             const std::pair<glow::Placeholder *, glow::Tensor *> ioPlaceholders,
             glow::Tensor &inputT,
             const std::vector<std::float_t> &inputData,
             std::vector<std::float_t> &outputData,
             std::size_t nFiles,
             std::size_t fileSize)
{
    glow::Placeholder *inputPH{ioPlaceholders.first};
    glow::Tensor *outputT{ioPlaceholders.second};

    auto inputIterData = inputData.cbegin();
    for (std::size_t ii = 0; ii < nFiles; ++ii) {
        auto inputIterTensor = inputT.getHandle().begin();
        auto outputIterTensor = outputT->getHandle().begin();

        for (std::size_t jj = 0; jj < fileSize; ++jj)
            *(inputIterTensor++) = *(inputIterData++);

        glow::updateInputPlaceholders(ioBindings, {inputPH}, {&inputT});
        loader.runInference(ioBindings, 1);
        outputData.push_back(*outputIterTensor);
    }
}

static void
writeOutputData(const std::vector<std::float_t> &outputData, const std::vector<std::string> files)
{
    if (writeOutput) {
        std::ofstream outputFile{};
        std::ostream *outputStream{};
        if (logFile.size() > 0) {
            outputFile.open(logFile.c_str());
            if (!outputFile) {
                std::cerr << "Unable to open output file: " << logFile << std::endl;
                return;
            }
            outputStream = &outputFile;
        } else
            outputStream = &std::cout;

        for (std::size_t jj = 0; jj < files.size(); ++jj)
            *outputStream << "[" << files[jj] << "] >>> " << outputData[jj] << std::endl;
    }
}

int 
main(int argc, char **argv)
{
    glow::PlaceholderBindings ioBindings{};
    std::pair<glow::Placeholder *, glow::Tensor *> ioPlaceholders{};
    glow::Loader loader(argc, argv);
    std::vector<std::size_t> dims{};
    std::vector<std::float_t> inputData{};
    std::vector<std::float_t> outputData{};
    std::vector<std::string> files{};
    glow::Tensor inputT{};
    std::size_t inputSize{};

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

    inputSize = gatherFiles(files);
    loadInputData(files, inputSize, inputData);
    runInference(loader, ioBindings, ioPlaceholders, inputT, inputData, outputData, files.size(), inputSize);
    writeOutputData(outputData, files);

    if (glow::profilingGraph())
        loader.generateAndSerializeQuantizationInfos(ioBindings);

    return 0;
}
