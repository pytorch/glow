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

#include "Loader.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/Support/CommandLine.h"

using namespace glow;

namespace {

/// Model compiler options.
llvm::cl::OptionCategory modelCompilerCat("Model Compiler Options");

llvm::cl::list<std::string> modelInputs(
    "model-input", llvm::cl::ZeroOrMore,
    llvm::cl::desc(
        "For ONNX models the inputs of the graph can be inferred   \n"
        "automatically and hence this option is not mandatory.     \n"
        "For Caffe2 models the graph definition does not contain   \n"
        "the description of the inputs and hence must be provided  \n"
        "explicitly using this option. One or more model inputs    \n"
        "are provided using the following format:                  \n"
        "   -model-input=<inputName1>,<inputType1>,<inputShape1>   \n"
        "   -model-input=<inputName2>,<inputType2>,<inputShape2>   \n"
        "   ....................................................   \n"
        "For quantized types the format is slightly different since\n"
        "the scale and offset parameters should also be provided:  \n"
        "   -model-input=<name>,<type>,<scale>,<offset>,<shape>    \n"
        "For example we can can provide one or more inputs:        \n"
        "   -model-input=input_03_data,float,[1]                   \n"
        "   -model-input=data_bias,int32,[1,32,32]                 \n"
        "   -model-input=data,int8q,0.123,-13,[1,10]               \n"
        "The supported types are:                                  \n"
        "   - float, float16 (floating point types)                \n"
        "   - int32, int64 (integer types)                         \n"
        "   - int8q, int16q, int32q (integer quantized types)      \n"
        "   - bool (logic type)\n"),
    llvm::cl::value_desc("name,type,shape"), llvm::cl::cat(modelCompilerCat));

} // namespace

/// Parse the strings and get the model input names and types.
static void getModelInputs(std::vector<std::string> &inputNames,
                           std::vector<Type> &inputTypes) {
  for (const auto &str : modelInputs) {
    // Split string. Expected format:
    // - <name>,<type>,<shape> for non-quantized type.
    // - <name>,<type>,<scale>,<offset>,<shape> for quantized type.
    auto strPair = llvm::StringRef(str).split(',');
    llvm::StringRef name = strPair.first;
    strPair = strPair.second.split(',');
    llvm::StringRef type = strPair.first;
    CHECK(name.size()) << "Model input name empty";
    CHECK(type.size()) << "Model input type empty";

    // Parse type string.
    ElemKind kind;
    if (type.equals("float")) {
      kind = ElemKind::FloatTy;
    } else if (type.equals("float16")) {
      kind = ElemKind::Float16Ty;
    } else if (type.equals("int8q")) {
      kind = ElemKind::Int8QTy;
    } else if (type.equals("int16q")) {
      kind = ElemKind::Int16QTy;
    } else if (type.equals("int32q")) {
      kind = ElemKind::Int32QTy;
    } else if (type.equals("int32")) {
      kind = ElemKind::Int32ITy;
    } else if (type.equals("int64")) {
      kind = ElemKind::Int64ITy;
    } else if (type.equals("bool")) {
      kind = ElemKind::BoolTy;
    } else {
      LOG(FATAL) << strFormat("Model input type %s not supported",
                              std::string(type).c_str());
    }

    // For quantized type get scale and offset.
    double scale;
    int32_t offset;
    if (isQuantizedElemKind(kind)) {
      strPair = strPair.second.split(',');
      CHECK(strPair.first.size()) << "Model input scale empty";
      CHECK(!strPair.first.getAsDouble(scale))
          << "Model input scale parameter invalid";
      strPair = strPair.second.split(',');
      CHECK(strPair.first.size()) << "Model input offset empty";
      CHECK(!strPair.first.getAsInteger(0, offset))
          << "Model input offset parameter invalid";
    }

    // Parse shape string.
    llvm::StringRef shape = strPair.second;
    CHECK(shape.size()) << "Model input shape empty";
    ShapeVector dims;
    CHECK_EQ(shape.front(), '[') << "First shape char should be [";
    shape = shape.drop_front();
    CHECK_EQ(shape.back(), ']') << "First shape char should be ]";
    shape = shape.drop_back();
    CHECK(shape.size()) << "Model input shape empty";
    size_t val;
    while (shape.contains(',')) {
      auto splitRes = shape.split(',');
      CHECK(!splitRes.first.getAsInteger(0, val))
          << "Model input shape integer invalid";
      dims.push_back(val);
      shape = splitRes.second;
    }
    CHECK(!shape.getAsInteger(0, val)) << "Model input shape integer invalid";
    dims.push_back(val);

    // Push data.
    inputNames.push_back(name);
    if (isQuantizedElemKind(kind)) {
      inputTypes.push_back(Type(kind, dims, (float)scale, offset));
    } else {
      inputTypes.push_back(Type(kind, dims));
    }
  }
}

int main(int argc, char **argv) {

  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  // Initialize loader.
  Loader loader;

  // Get model input names and types.
  std::vector<std::string> inputNames;
  std::vector<Type> inputTypes;
  getModelInputs(inputNames, inputTypes);

  // Emit bundle flag should be true.
  CHECK(emittingBundle())
      << "Bundle output directory not provided. Use the -emit-bundle option!";

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  if (!loader.getCaffe2NetDescFilename().empty()) {
    // For Caffe2 format the input placeholder names/types
    // must be provided explicitly.
    std::vector<const char *> inputNameRefs;
    std::vector<TypeRef> inputTypeRefs;
    for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {
      inputNameRefs.push_back(inputNames[idx].c_str());
      inputTypeRefs.push_back(&inputTypes[idx]);
    }
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        inputNameRefs, inputTypeRefs, *loader.getFunction()));
  } else {
    // For ONNX format the input placeholders names/types are
    // derived automatically.
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {}, {},
                                 *loader.getFunction()));
  }

  // Compile the model and generate the bundle.
  CompilationContext ctx;
  loader.compile(ctx);

  return 0;
}
