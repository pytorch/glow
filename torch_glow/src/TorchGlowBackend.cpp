// Copyright 2004-present Facebook. All Rights Reserved.
#include "TorchGlowBackend.h"
#include "FuseKnownPatterns.h"
#include "GlowCompileSpec.h"
#include "GlowFuser.h"
#include "InputMeta.h"
#include "Registration.h"

#include "glow/Runtime/ErrorReporter.h"

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>

#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace glow {

namespace {
void registerDummyOperatorWithSchema(const char *schema) {
  static std::unordered_set<c10::OperatorName> registered_ops;
  torch::jit::Operator op(
      schema,
      [](const torch::jit::Node *node) -> torch::jit::Operation {
        return [node](torch::jit::Stack *stack) {
          LOG(FATAL) << "Operator \"" << (*node)
                     << "\" has no implementation and is meant only as a "
                        "placeholder while fusing ops to run with Glow";
        };
      },
      at::AliasAnalysisKind::PURE_FUNCTION);
  if (registered_ops.count(op.schema().operator_name()) == 0) {
    torch::jit::RegisterOperators op_to_register({op});
    registered_ops.insert(op.schema().operator_name());
  }
}

void registerGlowHelperOps() {
  const char *conv2dSymbol =
      "glow::unpacked_quantized_conv2d(Tensor a_quant, Tensor w_quant, Tensor "
      "b, int[] stride, int[] padding, int[] dilation, int groups, float "
      "r_scale, int r_zero_point) -> Tensor";
  registerDummyOperatorWithSchema(conv2dSymbol);

  const char *conv2dReluSymbol =
      "glow::unpacked_quantized_conv2d_relu(Tensor a_quant, Tensor w_quant, "
      "Tensor "
      "b, int[] stride, int[] padding, int[] dilation, int groups, float "
      "r_scale, int r_zero_point) -> Tensor";
  registerDummyOperatorWithSchema(conv2dReluSymbol);

  const char *linearSymbol =
      "glow::unpacked_quantized_linear(Tensor a_quant, Tensor w_quant, Tensor "
      "b, float r_scale, int r_zero_point) -> Tensor";
  registerDummyOperatorWithSchema(linearSymbol);

  const char *conv3dSymbol =
      "glow::unpacked_quantized_conv3d(Tensor a_quant, Tensor w_quant, Tensor "
      "b, int[] stride, int[] padding, int[] dilation, int groups, float "
      "r_scale, int r_zero_point) -> Tensor";
  registerDummyOperatorWithSchema(conv3dSymbol);

  const char *conv3dReluSymbol =
      "glow::unpacked_quantized_conv3d_relu(Tensor a_quant, Tensor w_quant, "
      "Tensor  b, int[] stride, int[] padding, int[] dilation, int groups, "
      "float r_scale, int r_zero_point) -> Tensor";
  registerDummyOperatorWithSchema(conv3dReluSymbol);
}

/// Checks that all inputs are tensors and all outputs are either tensors or a
/// single tuple or list of tensors. \returns an error if any of these are
/// false otherwise returns the GraphOutputType describing the output.
Expected<GraphOutputType>
checkGraphInputsAndOutputs(const torch::jit::Graph &graph) {
  const auto inputs = graph.inputs();
  const auto outputs = graph.outputs();

  // Start at 1 since 0 is always self
  for (auto i = 1; i < inputs.size(); ++i) {
    const auto *in = inputs[i];
    const auto inKind = in->type()->kind();
    if (inKind != torch::jit::TypeKind::TensorType) {
      return MAKE_ERR(
          strFormat("Glow only accepts Tensor inputs but %dth input is a %s",
                    int(i), c10::typeKindToString(inKind)));
    }
  }

  // If there is only one output check that it's a tensor or a tuple or list of
  // tensors.
  if (outputs.size() == 1) {
    const auto outTy = outputs[0]->type();
    if (outTy->kind() == torch::jit::TypeKind::TensorType) {
      return GraphOutputType::TENSORS;
    } else if (outTy->kind() == torch::jit::TypeKind::TupleType) {
      const auto containedTys = outTy->containedTypes();
      for (auto i = 0; i < containedTys.size(); ++i) {
        RETURN_ERR_IF_NOT(
            containedTys[i]->kind() == torch::jit::TypeKind::TensorType,
            strFormat("Expected tuple output to contain only tensors but "
                      "element at position %i is a %s",
                      int(i), c10::typeKindToString(containedTys[i]->kind())));
      }
      return GraphOutputType::TENSOR_TUPLE;
    } else if (outTy->kind() == torch::jit::TypeKind::TensorType) {
      const auto containedTys = outTy->containedTypes();
      for (auto i = 0; i < containedTys.size(); ++i) {
        RETURN_ERR_IF_NOT(
            containedTys[i]->kind() == torch::jit::TypeKind::TensorType,
            strFormat("Expected list output to contain only tensors but "
                      "element at position %i is a %s",
                      int(i), c10::typeKindToString(containedTys[i]->kind())));
      }
      return GraphOutputType::TENSOR_LIST;
    } else {
      return MAKE_ERR(strFormat("Found unsupported output kind %s",
                                c10::typeKindToString(outTy->kind())));
    }
  }

  // For multiple output graphs, check that all outputs are tensors.
  for (auto i = 0; i < outputs.size(); ++i) {
    const auto outTy = outputs[i]->type();
    RETURN_ERR_IF_NOT(
        outTy->kind() == torch::jit::TypeKind::TensorType,
        strFormat("Expected multi-output graph to have only tensor outputs but "
                  "output at position %i is a %s",
                  int(i), c10::typeKindToString(outTy->kind())));
  }

  return GraphOutputType::TENSORS;
}

Error checkForFatalError(Error err) {
  if (!err || !err.peekErrorValue()->isFatalError()) {
    return err;
  }

  std::string msg = ERR_TO_STRING(std::move(err));
  if (auto reporters = ErrorReporterRegistry::ErrorReporters()) {
    reporters->report(msg);
  }
  LOG(FATAL) << "Non-recoverable device error: " << msg;
}

} // namespace

torch::jit::backend<TorchGlowBackend> &torchGlowBackend() {
  static auto cls = torch::jit::backend<TorchGlowBackend>("glow");
  return cls;
}

void registerTorchGlowBackendAndDeps() {
  (void)torchGlowBackend();
  registerPyTorchGlowCustomClasses();
  registerGlowHelperOps();
}

/// Unpacks conv2d and linear packed parameters and replaces
/// calls to quantized::conv2d with glow::unpacked_quantized_conv2d
/// which takes unpacked quantized arguments
static void
RewriteQuantPackedParamOps(std::shared_ptr<torch::jit::Graph> &graph) {
  // Quantized Conv2d pattern
  std::string quantized_conv2d_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        %res = quantized::conv2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%res))IR";
  std::string glow_quantized_conv2d_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
    %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)
    %stride : int[] = quantized::conv2d_stride(%packed_params)
    %padding : int[] = quantized::conv2d_padding(%packed_params)
    %dilation : int[] = quantized::conv2d_dilation(%packed_params)
    %groups : int = quantized::conv2d_groups(%packed_params)
    %res = glow::unpacked_quantized_conv2d(%a_quant, %w_quant, %b, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
    return (%res))IR";

  torch::jit::SubgraphRewriter quantized_conv2d_rewriter;
  quantized_conv2d_rewriter.RegisterRewritePattern(
      quantized_conv2d_pattern, glow_quantized_conv2d_pattern);
  quantized_conv2d_rewriter.runOnGraph(graph);

  // Quantized Conv2d + Relu pattern
  std::string quantized_conv2d_relu_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        %res = quantized::conv2d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%res))IR";
  std::string glow_quantized_conv2d_relu_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
    %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)
    %stride : int[] = quantized::conv2d_stride(%packed_params)
    %padding : int[] = quantized::conv2d_padding(%packed_params)
    %dilation : int[] = quantized::conv2d_dilation(%packed_params)
    %groups : int = quantized::conv2d_groups(%packed_params)
    %res = glow::unpacked_quantized_conv2d_relu(%a_quant, %w_quant, %b, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
    return (%res))IR";

  torch::jit::SubgraphRewriter quantized_conv2d_relu_rewriter;
  quantized_conv2d_relu_rewriter.RegisterRewritePattern(
      quantized_conv2d_relu_pattern, glow_quantized_conv2d_relu_pattern);
  quantized_conv2d_relu_rewriter.runOnGraph(graph);

  // Quantized Conv3d pattern
  std::string quantized_conv3d_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        %res = quantized::conv3d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%res))IR";
  std::string glow_quantized_conv3d_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
    %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)
    %stride : int[] = quantized::conv3d_stride(%packed_params)
    %padding : int[] = quantized::conv3d_padding(%packed_params)
    %dilation : int[] = quantized::conv3d_dilation(%packed_params)
    %groups : int = quantized::conv3d_groups(%packed_params)
    %res = glow::unpacked_quantized_conv3d(%a_quant, %w_quant, %b, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
    return (%res))IR";

  torch::jit::SubgraphRewriter quantized_conv3d_rewriter;
  quantized_conv3d_rewriter.RegisterRewritePattern(
      quantized_conv3d_pattern, glow_quantized_conv3d_pattern);
  quantized_conv3d_rewriter.runOnGraph(graph);

  // Quantized Conv3d + Relu pattern
  std::string quantized_conv3d_relu_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        %res = quantized::conv3d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%res))IR";
  std::string glow_quantized_conv3d_relu_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
    %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)
    %stride : int[] = quantized::conv3d_stride(%packed_params)
    %padding : int[] = quantized::conv3d_padding(%packed_params)
    %dilation : int[] = quantized::conv3d_dilation(%packed_params)
    %groups : int = quantized::conv3d_groups(%packed_params)
    %res = glow::unpacked_quantized_conv3d_relu(%a_quant, %w_quant, %b, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
    return (%res))IR";

  torch::jit::SubgraphRewriter quantized_conv3d_relu_rewriter;
  quantized_conv3d_relu_rewriter.RegisterRewritePattern(
      quantized_conv3d_relu_pattern, glow_quantized_conv3d_relu_pattern);
  quantized_conv3d_relu_rewriter.runOnGraph(graph);

  // Quantized Linear pattern
  std::string quantized_linear_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        %res = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%res))IR";
  std::string glow_quantized_linear_pattern = R"IR(
  graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
    %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
    %res = glow::unpacked_quantized_linear(%a_quant, %w_quant, %b, %r_scale, %r_zero_point)
    return (%res))IR";

  torch::jit::SubgraphRewriter quantized_linear_rewriter;
  quantized_linear_rewriter.RegisterRewritePattern(
      quantized_linear_pattern, glow_quantized_linear_pattern);
  quantized_linear_rewriter.runOnGraph(graph);
}

template <int DIMS>
static bool hasConvUnpackQParamUser(const torch::jit::Node *node) {
  static_assert(DIMS == 2 || DIMS == 3, "Only 2d and 3d conv supported");
  const char *convType = DIMS == 2 ? "conv2d" : "conv3d";

  static std::unordered_set<torch::jit::Symbol> unpackQuantNodeKinds = {
      torch::jit::Symbol::fromQualString(
          strFormat("quantized::%s_unpack", convType)),
  };

  const auto uses = node->output()->uses();
  if (uses.empty()) {
    return false;
  }

  for (const auto &u : uses) {
    const auto userKind = u.user->kind();
    if (unpackQuantNodeKinds.count(userKind)) {
      DCHECK_EQ(uses.size(), 5)
          << strFormat("Expected %s packed quantization parameters "
                       "to be used by exactly 5 unpacking nodes",
                       convType);
      return true;
    }
  }

  return false;
}

static void overrideTensorGradient(at::Tensor &t) {
  if (t.requires_grad()) {
    t = t.detach();
    t.set_requires_grad(false);
  }
}

template <int DIMS>
static void processConvPackedQParams(torch::jit::Graph &graph,
                                     const c10::IValue &ival,
                                     torch::jit::Node *paramsNode) {
  static_assert(DIMS == 2 || DIMS == 3, "Only 2d and 3d conv supported");
  const char *convType = DIMS == 2 ? "conv2d" : "conv3d";
  auto packed_params = ival.toCustomClass<ConvPackedParamsBase<DIMS>>();
  torch::jit::WithInsertPoint guard(paramsNode);
  const auto uses = paramsNode->output()->uses();
  for (const auto &u : uses) {
    auto node = u.user;
    const auto userKind = node->kind();
    if (userKind == torch::jit::Symbol::fromQualString(
                        strFormat("quantized::%s_unpack", convType))) {
      at::Tensor ptWeightTensor;
      c10::optional<at::Tensor> ptBiasTensorTmp;
      std::tie(ptWeightTensor, ptBiasTensorTmp) = packed_params->unpack();
      overrideTensorGradient(ptWeightTensor);
      torch::jit::Value *paramConst =
          torch::jit::insertConstant(graph, ptWeightTensor);
      node->outputs().at(0)->replaceAllUsesWith(paramConst);

      if (ptBiasTensorTmp.has_value()) {
        at::Tensor &t = ptBiasTensorTmp.value();
        overrideTensorGradient(t);
        paramConst = torch::jit::insertConstant(graph, t);
        node->outputs().at(1)->replaceAllUsesWith(paramConst);
      } else {
        at::Tensor t = at::zeros(ptWeightTensor.size(0));
        paramConst = torch::jit::insertConstant(graph, t);
        node->outputs().at(1)->replaceAllUsesWith(paramConst);
      }
    } else if (node->kind() == torch::jit::Symbol::fromQualString(strFormat(
                                   "quantized::%s_stride", convType))) {
      torch::List<int64_t> stride;
      stride = packed_params->stride();
      torch::jit::Value *paramConst = torch::jit::insertConstant(graph, stride);
      node->outputs().at(0)->replaceAllUsesWith(paramConst);
    } else if (node->kind() == torch::jit::Symbol::fromQualString(strFormat(
                                   "quantized::%s_padding", convType))) {
      torch::List<int64_t> pad;
      pad = packed_params->padding();
      torch::jit::Value *paramConst = torch::jit::insertConstant(graph, pad);
      node->outputs().at(0)->replaceAllUsesWith(paramConst);
    } else if (node->kind() == torch::jit::Symbol::fromQualString(strFormat(
                                   "quantized::%s_dilation", convType))) {
      torch::List<int64_t> dilation;
      dilation = packed_params->dilation();
      torch::jit::Value *paramConst =
          torch::jit::insertConstant(graph, dilation);
      node->outputs().at(0)->replaceAllUsesWith(paramConst);
    } else if (node->kind() == torch::jit::Symbol::fromQualString(strFormat(
                                   "quantized::%s_groups", convType))) {
      int64_t groups;
      groups = packed_params->groups();
      torch::jit::Value *paramConst = torch::jit::insertConstant(graph, groups);
      node->outputs().at(0)->replaceAllUsesWith(paramConst);
    }
  }
}

static bool hasLinearUnpackQParamUser(const torch::jit::Node *node) {
  static std::unordered_set<torch::jit::Symbol> unpackQuantNodeKinds = {
      torch::jit::Symbol::fromQualString("quantized::linear_unpack"),
  };

  const auto uses = node->output()->uses();
  if (uses.empty()) {
    return false;
  }

  const auto userKind = uses[0].user->kind();

  if (unpackQuantNodeKinds.count(userKind)) {
    DCHECK_EQ(uses.size(), 1) << "Expected unpacked quantization parameters "
                                 "to only be used by one node";
    return true;
  }

  return false;
}

static Error processLinearPackedQParams(torch::jit::Graph &graph,
                                        const c10::IValue &ival,
                                        torch::jit::Node *paramsNode) {
  auto packed_params = ival.toCustomClass<LinearPackedParamsBase>();
  torch::jit::WithInsertPoint guard(paramsNode);
  auto node = paramsNode->output()->uses()[0].user;
  const auto userKind = node->kind();
  if (userKind ==
      torch::jit::Symbol::fromQualString("quantized::linear_unpack")) {
    at::Tensor ptWeightTensor;
    c10::optional<at::Tensor> ptBiasTensorTmp;
    std::tie(ptWeightTensor, ptBiasTensorTmp) = packed_params->unpack();
    overrideTensorGradient(ptWeightTensor);
    torch::jit::Value *paramConst =
        torch::jit::insertConstant(graph, ptWeightTensor);
    node->outputs().at(0)->replaceAllUsesWith(paramConst);
    if (ptBiasTensorTmp.has_value()) {
      at::Tensor &t = ptBiasTensorTmp.value();
      overrideTensorGradient(t);
      paramConst = torch::jit::insertConstant(graph, t);
      node->outputs().at(1)->replaceAllUsesWith(paramConst);
    } else { // TODO Handle bias-not-exists case
      return MAKE_ERR("Preprocess for empty bias is not yet supported.");
    }
  }
  return Error::success();
}

static Error ProcessPackedParams(torch::jit::Graph &graph,
                                 torch::jit::IValue module) {
  // Map from the Value in the Graph of an ivalue::Object to the Object and a
  // string representing it's place in the module hierarchy.
  std::unordered_map<const torch::jit::Value *,
                     std::pair<const c10::ivalue::Object *, std::string>>
      objectTree;

  // Load graph inputs that are Objects.
  auto graphInputValues = graph.inputs();
  const auto &object = module.toObjectRef();
  // Only the first graph input (self) is expected to be an object.
  objectTree[graphInputValues[0]] =
      std::make_pair(&object, object.type()->str().c_str());

  // Load prim::GetAttr nodes.
  for (const auto &node : graph.nodes()) {
    if (node->kind() != torch::jit::prim::GetAttr) {
      continue;
    }

    RETURN_IF_ERR(PyTorchModelLoader::checkInputAndOutputSizes(
        node->inputs(), 1, node->outputs(), 1));

    const auto *inputValue = node->input();
    const auto *outputValue = node->output();

    RETURN_ERR_IF_NOT(objectTree.count(inputValue),
                      "Missing input for prim::getAttr");

    const auto &parent = objectTree.at(inputValue);
    const auto *parentObject = parent.first;

    const auto attrName = node->s(torch::jit::attr::name);
    const auto ival = parentObject->getAttr(attrName);

    // Concatenation of names of Objects and fields referenced in the Module
    // tree.
    const auto &nameHierarchy = parent.second;
    const auto newNameHierarchy =
        strFormat("%s_%s", nameHierarchy.c_str(), attrName.c_str());

    if (ival.isObject()) {
      if (hasConvUnpackQParamUser<2>(node)) {
        processConvPackedQParams<2>(graph, ival, node);
      } else if (hasConvUnpackQParamUser<3>(node)) {
        processConvPackedQParams<3>(graph, ival, node);
      } else if (hasLinearUnpackQParamUser(node)) {
        RETURN_IF_ERR(processLinearPackedQParams(graph, ival, node));
      } else {
        objectTree[outputValue] =
            std::make_pair(&ival.toObjectRef(), newNameHierarchy);
      }
    }
  }
  return Error::success();
}

c10::IValue
TorchGlowBackend::preprocess(c10::IValue mod,
                             c10::impl::GenericDict method_compile_spec) {
  // We do nothing in the preprocess, instead we do them in compile()
  return mod;
}

Error applySettingsOverrideFlagsToPyTorchLoaderSettings(
    PyTorchLoaderSettings &settings) {
  // TODO:
  return Error::success();
}

Error applyCompilationGroupSettingsToPyTorchLoaderSettings(
    PyTorchLoaderSettings &settings,
    const CompilationGroupSettings &newSettings) {
  if (newSettings.num_devices_to_use == -1) {
    settings.saturateHost = true;
  } else {
    return MAKE_ERR("Only num_devices_to_use=-1 supported currently");
  }

  settings.replicationCount = newSettings.replication_count;
  settings.backendSpecificOpts = newSettings.backend_specific_opts;
  settings.convertToFP16 = newSettings.convert_to_fp16;

  // Ensure override flags are honored
  RETURN_IF_ERR(applySettingsOverrideFlagsToPyTorchLoaderSettings(settings));
  return Error::success();
}

Error applyCompilationSpecSettingsToPyTorchLoaderSettings(
    PyTorchLoaderSettings &settings,
    const CompilationSpecSettings &newSettings) {
  settings.backendName = newSettings.glow_backend;
  settings.enableDebugFuser = newSettings.enable_fuser;

  // Ensure override flags are honored
  RETURN_IF_ERR(applySettingsOverrideFlagsToPyTorchLoaderSettings(settings));
  return Error::success();
}

Error applyFuserSettingsToPyTorchLoaderSettings(
    PyTorchLoaderSettings &settings, const FuserSettings &newSettings) {
  if (newSettings.min_fusion_group_size > 0) {
    settings.minFusionGroupSize = newSettings.min_fusion_group_size;
  }

  if (newSettings.max_fusion_merge_size > 0) {
    settings.maxFusionMergeSize = newSettings.max_fusion_merge_size;
  }

  settings.fusionStartIndex = newSettings.fusion_start_index;
  settings.fusionEndIndex = newSettings.fusion_end_index;
  for (const auto &symbol : newSettings.op_blacklist) {
    settings.opBlacklist.insert(torch::jit::Symbol::fromQualString(symbol));
  }

  // Ensure override flags are honored
  RETURN_IF_ERR(applySettingsOverrideFlagsToPyTorchLoaderSettings(settings));
  return Error::success();
}

/// Implementation of to_backend compile method for Glow. \returns a map from
/// function name in the preprocessed module to the CachingGraphRunner or
/// JITGraphRunner depending on settings for each method or if an
/// error occurs then returns and Error which is converted to an exception for
/// handling within PyTorch.
static Expected<std::unordered_map<
    std::string, std::pair<std::unique_ptr<CachingGraphRunner>,
                           std::unique_ptr<JITGraphRunner>>>>
compileImpl(const torch::jit::Module &origModule,
            const c10::impl::GenericDict &method_compile_spec) {

  std::unordered_map<std::string, std::pair<std::unique_ptr<CachingGraphRunner>,
                                            std::unique_ptr<JITGraphRunner>>>
      methodToRunnerMap;
  std::unordered_map<std::string, std::shared_ptr<torch::jit::Graph>>
      nameToOrigGraph;

  // Check input and outputs types of each method and inline and remove profiled
  // shapes (this must happen before other optimizations)
  for (const auto &kv : method_compile_spec) {
    const auto &methodName = kv.key().toStringRef();
    auto method = origModule.get_method(methodName);
    auto graph = method.graph();

    GraphOutputType graphOutputType;
    ASSIGN_VALUE_OR_RETURN_ERR(graphOutputType,
                               checkGraphInputsAndOutputs(*graph));

    // Output lists no supported yet
    if (graphOutputType == GraphOutputType::TENSOR_LIST) {
      return MAKE_ERR("Tensor list output not supported.");
    }

    torch::jit::Inline(*graph);
    torch::jit::ClearProfilingInformation(graph);
    torch::jit::EraseShapeInformation(graph);
    nameToOrigGraph[methodName] = graph->copy();
    /* Note: The following LowerAllTuples call is a bit of a hack. The
     * GraphExecutor that takes this copy of the graph would otherwise not have
     * the tuples lowered, which would cause an issue, given that the two
     * versions would have different output types.
     */
    LowerAllTuples(nameToOrigGraph[methodName]);
  }

  // JIT graph optimizations before freezing
  for (const auto &kv : method_compile_spec) {
    const auto &methodName = kv.key().toStringRef();
    auto method = origModule.get_method(methodName);
    auto graph = method.graph();

    RewriteQuantPackedParamOps(graph);
    RETURN_IF_ERR(ProcessPackedParams(*graph, origModule._ivalue()));
  }

  // Freeze
  auto frozenModule = torch::jit::freeze_module(origModule);

  // Cleanup JIT graphs
  for (const auto &kv : method_compile_spec) {
    const auto &methodName = kv.key().toStringRef();
    auto method = frozenModule.get_method(methodName);
    auto graph = method.graph();

    torch::jit::RemoveListMutation(graph);
    torch::jit::RemoveTensorMutation(graph);

    detail::fuseConcat(graph);
    torch::jit::CanonicalizeOps(graph);
    EliminateCommonSubexpression(graph);
    ConstantPooling(graph);
    // EliminateDeadCode should be last
    EliminateDeadCode(graph);
  }

  auto compileModule = frozenModule.clone();
  // Compile each method
  for (const auto &kv : method_compile_spec) {
    const auto methodName = kv.key().toString()->string();
    const auto &method = compileModule.get_method(methodName);
    auto it = nameToOrigGraph.find(methodName);
    CHECK(it != nameToOrigGraph.end())
        << "Cannot find corresponding original graph for graph: " << methodName;
    auto origGraph = it->second;

    const CompilationSpec &spec = *kv.value().toCustomClass<CompilationSpec>();
    RETURN_IF_ERR(spec.validate());

    PyTorchLoaderSettings baseSettings =
        getGlobalPyTorchLoaderSettingsSnapshot();

    // Apply settings from CompilationSpecSettings
    RETURN_IF_ERR(applyCompilationSpecSettingsToPyTorchLoaderSettings(
        baseSettings, *spec.settings));

    // Apply settings from FuserSettings
    RETURN_IF_ERR(applyFuserSettingsToPyTorchLoaderSettings(
        baseSettings, *spec.fuser_settings));

    // Apply default CompilationGroupSettings
    RETURN_IF_ERR(applyCompilationGroupSettingsToPyTorchLoaderSettings(
        baseSettings, *spec.default_compilation_group_settings));

    // Override settings from gflags
    RETURN_IF_ERR(
        applySettingsOverrideFlagsToPyTorchLoaderSettings(baseSettings));

    auto graph = method.function().graph();
    graph = graph->copy();

    LowerAllTuples(graph);

    if (baseSettings.enableDebugFuser) {
      LOG(WARNING) << "TorchGlowBackend using GlowFuser";

      // Run fusion flow using JIT graph runner
      std::unique_ptr<JITGraphRunner> runner = std::make_unique<JITGraphRunner>(
          compileModule._ivalue(), graph, baseSettings);
      methodToRunnerMap.emplace(methodName,
                                std::make_pair(nullptr, std::move(runner)));
    } else {
      LOG(INFO) << "TorchGlowBackend using CachingGraphRunner";
      // Remove "self" input
      RETURN_ERR_IF_NOT(graph->block()->inputs()[0]->uses().empty(),
                        "self must have no uses in order to lower to Glow.");
      graph->block()->eraseInput(0);

      // Create a corresponding runner and store {handle, runner} pair.
      std::unique_ptr<CachingGraphRunner> runner =
          std::make_unique<glow::CachingGraphRunner>(
              graph,
              glow::getHostManager(baseSettings.backendName,
                                   baseSettings.numDevices),
              baseSettings, /*useRunOnly*/ true, origGraph,
              origModule._ivalue());

      // Compile each compilation group
      for (const auto &compilationGroup : spec.compilation_groups) {
        // Apply CompilationGroupSettings settings
        auto compilationGroupSettings = baseSettings;
        RETURN_IF_ERR(applyCompilationGroupSettingsToPyTorchLoaderSettings(
            compilationGroupSettings, *compilationGroup->settings));
        // Compile each input set
        for (const auto &inputSet : compilationGroup->input_sets) {
          InputMetaStack metaStack = getInputMetas(inputSet);
          auto err = runner->warmCache(metaStack, compilationGroupSettings,
                                       /*loader*/ nullptr,
                                       /*useMaxSizeCompilation*/ false);
          err = checkForFatalError(std::move(err));
          RETURN_IF_ERR(err);
        }
      }
      methodToRunnerMap.emplace(methodName,
                                std::make_pair(std::move(runner), nullptr));
    }
  }

  return methodToRunnerMap;
}

c10::impl::GenericDict
TorchGlowBackend::compile(c10::IValue processed,
                          c10::impl::GenericDict method_compile_spec) {
  auto moduleCompile = processed.toModule();

  moduleCompile.eval();
  auto runnersOrErr = compileImpl(moduleCompile, method_compile_spec);
  if (!runnersOrErr) {
    auto err = runnersOrErr.takeError();
    err = checkForFatalError(std::move(err));
    throw std::runtime_error(ERR_TO_STRING(std::move(err)));
  }

  auto handles = c10::Dict<std::string, int64_t>();
  int64_t nextHandle = 0;

  // Backend is created on each to_backend call --> use simple
  // consecutive keys for methods.
  for (auto &methodNameAndRunner : *runnersOrErr) {
    handles.insert(methodNameAndRunner.first, nextHandle);
    handleToRunnerMap_.emplace(nextHandle,
                               std::move(methodNameAndRunner.second));
    nextHandle++;
  }

  return c10::impl::toGenericDict(handles);
}

c10::impl::GenericList
TorchGlowBackend::execute(c10::IValue handle, c10::impl::GenericList inputs) {

  auto it = handleToRunnerMap_.find(handle.toInt());
  if (it == handleToRunnerMap_.end()) {
    throw std::out_of_range("Could not find runner for handle " +
                            std::to_string(handle.toInt()));
  }

  const auto &runnerPair = it->second;

  torch::jit::Stack stack;

  Error err = glow::ErrorEmpty();
  if (runnerPair.first) {
    for (const auto &i : inputs) {
      torch::jit::push(stack, i);
    }
    err = it->second.first->run(stack);
  } else if (runnerPair.second) {
    stack = it->second.second->onExecute(inputs);
  } else {
    throw std::runtime_error("Could not any type of runner for handle");
  }

  if (err) {
    err = checkForFatalError(std::move(err));
    throw std::runtime_error(ERR_TO_STRING(std::move(err)));
  }

  c10::List<at::Tensor> outputList;
  for (const auto &value : torch::jit::last(stack, stack.size())) {
    outputList.emplace_back(value.toTensor());
  }
  return c10::impl::toList(outputList);
}

JITGraphRunner::JITGraphRunner(c10::IValue module,
                               std::shared_ptr<torch::jit::Graph> graph,
                               PyTorchLoaderSettings settings)
    : module_(module), graph_(graph), ptGraphExecutor_(graph, "forward"),
      settings_(std::move(settings)) {
  glow::registDefaultGlowFusionSymbolOnce();
  std::cout << "Running Glow Fusion Pass" << std::endl;
  std::unordered_set<torch::jit::NodeKind> supportedKinds;
  std::unordered_set<torch::jit::NodeKind> unsupportedKinds;
  for (auto node : graph_->nodes()) {
    auto nk = node->kind();
    if (supportedKinds.count(nk) == 0) {
      if (PyTorchModelLoader::isNodeSupported(node)) {
        supportedKinds.emplace(nk);
      } else if (unsupportedKinds.count(nk) == 0) {
        unsupportedKinds.emplace(nk);
      }
    }
  }
  if (unsupportedKinds.size() == 0) {
    std::cout << "No unsupported nodes detected." << std::endl;
  } else {
    std::cout << "Unsupported Nodes:" << std::endl;
    for (auto nk : unsupportedKinds) {
      std::cout << nk.toQualString() << std::endl;
    }
  }
  std::cout << "Supported Nodes:" << std::endl;
  for (auto nk : supportedKinds) {
    std::cout << nk.toQualString() << std::endl;
  }

  glowCustomFuse(graph_, settings_);

  // Print graph after fusion
  std::cout << "Graph after Glow fusion pass:\n"
            << "(" << countFusionNodes() << " fusion nodes)" << std::endl
            << *graph_ << std::endl;
}

torch::jit::Stack JITGraphRunner::onExecute(c10::impl::GenericList inputs) {
  CHECK_EQ(getGlobalPyTorchLoaderSettingsMutable().fusionPassEnabled, false);
  torch::jit::Stack stack;
  torch::jit::push(stack, module_);
  for (const auto &i : inputs) {
    torch::jit::push(stack, i);
  }
  ptGraphExecutor_.run(stack);
  return stack;
}

int JITGraphRunner::countFusionNodes() {
  int count = 0;
  for (auto node : graph_->nodes()) {
    if (node->kind() == getGlowSymbol()) {
      count++;
    }
  }
  return count;
}
} // namespace glow
