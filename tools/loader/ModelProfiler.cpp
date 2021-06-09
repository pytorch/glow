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
#include "LoaderUtils.h"
#include "glow/Base/TensorSerialization.h"
#include "llvm/Support/CommandLine.h"

using namespace glow;

llvm::cl::OptionCategory modelProfilerCat("Model Profiler Options");

namespace {
llvm::cl::list<std::string> inputDatasetOpts(
    "input-dataset", llvm::cl::ZeroOrMore,
    llvm::cl::desc(
        "Provide a dataset for a model input as a set of file paths by using \n"
        "this option with the following format:                              \n"
        "    -input-dataset=<name>,<format>,<source>,<opts>                  \n"
        "<name>   the name of the model input placeholder (tensor) where the \n"
        "         dataset files will be loaded during run-time.              \n"
        "<format> the format of all the files from the given dataset:        \n"
        "         - 'rawbin': raw binary format. Each binary file corresponds\n"
        "           to a tensor and contains the data serialized as a binary \n"
        "           blob without extra meta information (tensor data type or \n"
        "           shape) because the tensor is statically configured before\n"
        "           loading the data. The data is expected to be serialized  \n"
        "           with the correct size and layout as the tensor in which  \n"
        "           it will be loaded. For example, for a float32 tensor with\n"
        "           shape [2,3], the binary file is expected to have the size\n"
        "           2 x 3 x 4 (float32) = 24 bytes.                          \n"
        "         - 'rawtxt': raw text format. Each text file corresponds to \n"
        "           a tensor and contains data serialized as a linear list   \n"
        "           of comma separated values in text format without extra   \n"
        "           meta information (tensor data type or shape) because the \n"
        "           tensor is statically configured before loading the data. \n"
        "           The data is expected to be serialized with the correct   \n"
        "           size and layout as the tensor in which it will be loaded.\n"
        "           For example, for a float32 tensor with shape [2,3], the  \n"
        "           text file is expected to contain a list of 6 values      \n"
        "           separated by comma like this (extra spaces and newlines  \n"
        "           are allowed):                                            \n"
        "               1.0, 2.0, 3.0, 4.0, 5.0, 6.0,                        \n"
        "<source> specifies the dataset source:                              \n"
        "         - 'file': the dataset is specified as a text file which    \n"
        "           contains the relative or absolute paths of all the files \n"
        "           in the dataset, listed one per line, separated by comma  \n"
        "           or not. The path of the dataset file is given as the     \n"
        "           first argument in the <opts> list. If a second argument  \n"
        "           is given in the <opts> list (optional), that will be     \n"
        "           concatenated (prepended) to all the paths from the file. \n"
        "           The dataset file must contain only ONE PATH PER LINE.    \n"
        "           After the first comma or space character, the rest of the\n"
        "           line is ignored. All the examples below are valid:       \n"
        "               data0.bin                                            \n"
        "               data1.bin,                                           \n"
        "               data2.bin 'cat'                                      \n"
        "               data3.bin,dog                                        \n"
        "               data4.bin ,2                                         \n"
        "               data5.bin,1                                          \n"
        "           Do NOT use file paths which contain spaces.              \n"
        "         - 'dir': the dataset is specified as all the files from a  \n"
        "           given directory listed alphabetically. The directory path\n"
        "           is specified with the first argument in the <opts> list. \n"
        "           Make sure the directory does not contain other items than\n"
        "           the dataset files (folders, symlinks, etc).              \n"
        "<opts>   extra options dependent on the <source> field.             \n"
        "This option will be used for each of the model inputs.              \n"
        "\nExample 1:                                                        \n"
        "    -input-dataset=input1,rawbin,file,dataset.csv                   \n"
        "    The dataset paths for the 'input1' model input are read from the\n"
        "    'dataset.csv' file which could have the following content:      \n"
        "        /data_folder/data0.dat,                                     \n"
        "        /data_folder/data1.dat,                                     \n"
        "        .......................                                     \n"
        "    All the files listed are assumed to be in raw binary format.    \n"
        "\nExample 2:                                                        \n"
        "    -input-dataset=input2,rawbin,file,dataset.csv,/data_folder      \n"
        "    The dataset files for the 'input2' model input are read from the\n"
        "    'dataset.csv' file which could have the following content:      \n"
        "        data0.dat,                                                  \n"
        "        data1.dat,                                                  \n"
        "        ..........                                                  \n"
        "    All the file paths listed will be concatenated (prepended) with \n"
        "    the '/data_folder' base directory path when loading. All the    \n"
        "    files listed are assumed to be in raw binary format.            \n"
        "\nExample 3:                                                        \n"
        "    -input-dataset=input3,rawtxt,dir,/data_folder                   \n"
        "    The dataset files for the 'input3' model input are all the files\n"
        "    from the '/data_folder' directory listed alphabetically. The    \n"
        "    files are assumed to be in raw text format.\n"),
    llvm::cl::value_desc("name,format,source,opts"),
    llvm::cl::cat(modelProfilerCat));
} // namespace

/// Parse the 'input-dataset' option and get the arguments.
static void
getInputDatasets(std::vector<std::string> &inputNames,
                 std::vector<std::string> &inputFormats,
                 std::vector<std::string> &inputSources,
                 std::vector<std::vector<std::string>> &inputOptions) {
  for (const auto &str : inputDatasetOpts) {
    // Parse name.
    auto strPair = llvm::StringRef(str).split(',');
    llvm::StringRef name = strPair.first;
    checkCond(name.size(), "Model input name for dataset is empty!");
    inputNames.push_back(name.str());
    // Parse format.
    strPair = strPair.second.split(',');
    llvm::StringRef format = strPair.first;
    checkCond(format.size(),
              strFormat("Model input dataset format is empty for '%s'!",
                        name.data()));
    inputFormats.push_back(format.str());
    // Parse source.
    strPair = strPair.second.split(',');
    llvm::StringRef source = strPair.first;
    checkCond(source.size(),
              strFormat("Model input dataset source is empty for '%s'!",
                        name.data()));
    inputSources.push_back(source.str());
    // Parse options (optional).
    std::vector<std::string> options;
    while (strPair.second.size() != 0) {
      strPair = strPair.second.split(',');
      llvm::StringRef opt = strPair.first;
      checkCond(opt.size(),
                strFormat("Model input dataset options is empty for '%s'!",
                          name.data()));
      options.push_back(opt.str());
    }
    inputOptions.push_back(options);
  }
}

int main(int argc, char **argv) {

  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Dump profile option should be set.
  checkCond(profilingGraph(),
            "Use the 'dump-profile' option to specify the dump profile path!");

  // Get the input dataset options.
  std::vector<std::string> inputNames;
  std::vector<std::string> inputFormats;
  std::vector<std::string> inputSources;
  std::vector<std::vector<std::string>> inputOptions;
  getInputDatasets(inputNames, inputFormats, inputSources, inputOptions);
  auto numInputDatasets = inputNames.size();
  checkCond(numInputDatasets >= 1,
            "At least one input dataset must be specified using the "
            "'input-dataset' option!");

  // Get profiling dataset.
  std::vector<UnlabeledDataSet> inputDatasets(numInputDatasets);
  for (size_t idx = 0; idx < numInputDatasets; idx++) {
    auto inputName = inputNames[idx];
    auto inputSrc = inputSources[idx];
    auto inputOpts = inputOptions[idx];
    if (inputSrc == "file") {
      // Get dataset paths from file.
      if (inputOpts.size() == 1) {
        inputDatasets[idx] = readUnlabeledDataSetFromFile(inputOpts[0], "");
      } else if (inputOpts.size() == 2) {
        inputDatasets[idx] =
            readUnlabeledDataSetFromFile(inputOpts[0], inputOpts[1]);
      } else {
        exitWithErr(strFormat("Invalid number of parameters provided for the "
                              "dataset 'file' of the '%s' input!",
                              inputName.c_str()));
      }
    } else if (inputSrc == "dir") {
      // Get dataset paths from directory.
      if (inputOpts.size() == 1) {
        inputDatasets[idx] = readUnlabeledDataSetFromDir(inputOpts[0]);
      } else {
        exitWithErr(strFormat("Invalid number of parameters provided for the "
                              "dataset 'dir' of the '%s' input!",
                              inputName.c_str()));
      }
    } else {
      exitWithErr(strFormat("Input dataset source '%s' is not supported!",
                            inputSrc.c_str()));
    }
  }

  // Verify we have the same number of entries for all the datasets.
  size_t entryNum = inputDatasets[0].size();
  for (size_t idx = 1; idx < numInputDatasets; idx++) {
    checkCond(inputDatasets[idx].size() == entryNum,
              strFormat("The profiling dataset for the input '%s' does not "
                        "have the same number of entries as the other inputs!",
                        inputNames[idx].c_str()));
  }

  // Initialize the loader object.
  Loader loader;

  // Load the model.
  loader.loadModel();

  // Get the model input placeholders in the same order as the input dataset
  // options.
  auto inputVarsMapping = loader.getInputPlaceholderMap();
  auto modelNumInputs = inputVarsMapping.size();
  checkCond(modelNumInputs == numInputDatasets,
            "Not all the model inputs where provided with the 'input-dataset' "
            "parameter!");
  std::vector<Placeholder *> inputPlaceholders;
  for (const auto &name : inputNames) {
    auto it = inputVarsMapping.find(name);
    checkCond(
        it != inputVarsMapping.end(),
        strFormat("Name '%s' is not a model input placeholder!", name.c_str()));
    inputPlaceholders.push_back(it->second);
  }

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Get compilation options for profiling.
  CompilationContext cctx =
      loader.getCompilationContext(QuantizationMode::Profile);
  cctx.bindings = &bindings;

  // Compile the function.
  loader.compile(cctx);

  // Run profiling for all the dataset entries. The profiling information is
  // automatically aggregated for all the inference runs.
  for (size_t entryIdx = 0; entryIdx < entryNum; entryIdx++) {

    // Load tensor data.
    for (size_t inputIdx = 0; inputIdx < modelNumInputs; inputIdx++) {
      Tensor *inputTensor = bindings.get(inputPlaceholders[inputIdx]);
      std::string filePath = inputDatasets[inputIdx][entryIdx];
      std::string fileFormat = inputFormats[inputIdx];
      if (fileFormat == "rawbin") {
        TensorSerializationOptions opts;
        opts.withType = false;
        glow::loadTensorFromBinaryFile(*inputTensor, filePath.c_str(), opts);
      } else if (fileFormat == "rawtxt") {
        TensorSerializationOptions opts;
        opts.withType = false;
        glow::loadTensorFromTextFile(*inputTensor, filePath.c_str(), opts);
      } else {
        exitWithErr(strFormat("Input dataset format '%s' invalid!",
                              fileFormat.c_str()));
      }
    }

    // Run inference.
    loader.runInference(bindings, 1 /*batchSize*/);
  }

  // Dump the final profile.
  loader.generateAndSerializeProfilingInfos(bindings);

  return 0;
}
