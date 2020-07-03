// Copyright 2004-present Facebook. All Rights Reserved.
#include "TorchGlowBackend.h"
#include "GlowCompileSpec.h"
#include "Registration.h"
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace glow {

static std::vector<glow::InputMeta>
parseMethodCompileSpec(const GlowCompileSpec &method_spec) {
  setHostManager(method_spec.getBackend());
  std::vector<glow::InputMeta> inputMeta;
  for (const auto &in : method_spec.inputs()) {
    std::vector<glow::dim_t> dims;
    for (auto d : in.dims()) {
      dims.emplace_back(static_cast<size_t>(d));
    }
    inputMeta.emplace_back(in.type(), std::move(dims));
  }
  return inputMeta;
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
  std::string conv2dSymbol = "glow::unpacked_quantized_conv2d";
  c10::Symbol glowUnpackedConv2dSymbol =
      at::Symbol::fromQualString(conv2dSymbol);
  registerGlowOp(glowUnpackedConv2dSymbol);

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
  std::string conv2dReluSymbol = "glow::unpacked_quantized_conv2d_relu";
  c10::Symbol glowUnpackedConv2dReluSymbol =
      at::Symbol::fromQualString(conv2dReluSymbol);
  registerGlowOp(glowUnpackedConv2dReluSymbol);

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
  std::string conv3dSymbol = "glow::unpacked_quantized_conv3d";
  c10::Symbol glowUnpackedConv3dSymbol =
      at::Symbol::fromQualString(conv3dSymbol);
  registerGlowOp(glowUnpackedConv3dSymbol);

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
  std::string conv3dReluSymbol = "glow::unpacked_quantized_conv3d_relu";
  c10::Symbol glowUnpackedConv3dReluSymbol =
      at::Symbol::fromQualString(conv3dReluSymbol);
  registerGlowOp(glowUnpackedConv3dReluSymbol);

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
  std::string linearSymbol = "glow::unpacked_quantized_linear";
  c10::Symbol glowUnpackedLinearSymbol =
      at::Symbol::fromQualString(linearSymbol);
  registerGlowOp(glowUnpackedLinearSymbol);
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
      } else { // TODO Handle bias-not-exists case
        throw std::invalid_argument(
            "Preprocess for empty bias is not yet supported.");
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

static void processLinearPackedQParams(torch::jit::Graph &graph,
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
      throw std::invalid_argument(
          "Preprocess for empty bias is not yet supported.");
    }
  }
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
        processLinearPackedQParams(graph, ival, node);
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
  torch::jit::Module m = mod.toModule();
  m.eval();

  // Unpack qparams
  for (const auto &kv : method_compile_spec) {
    const auto &methodName = kv.key().toStringRef();
    auto method = m.get_method(methodName);
    auto graph = method.graph();
    torch::jit::Inline(*graph);
    RewriteQuantPackedParamOps(graph);
    glow::Error err = ProcessPackedParams(*graph, m._ivalue());
    if (static_cast<bool>(err)) {
      throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
    }
  }

  // Freeze
  m = torch::jit::freeze_module(m);

  // Cleanup JIT graphs
  for (const auto &kv : method_compile_spec) {
    const auto &methodName = kv.key().toStringRef();
    auto method = m.get_method(methodName);
    auto graph = method.graph();
    EliminateDeadCode(graph);
    EliminateCommonSubexpression(graph);
    ConstantPooling(graph);
  }

  return m._ivalue();
}

c10::impl::GenericDict
TorchGlowBackend::compile(c10::IValue processed,
                          c10::impl::GenericDict method_compile_spec) {
  auto module = processed.toModule();
  auto handles = c10::Dict<std::string, int64_t>();

  // Compile each method
  int64_t key = 0;
  for (const auto &method : module.get_methods()) {
    auto g = method.function().graph();
    // Remove "self" input
    CHECK(g->block()->inputs()[0]->uses().empty())
        << "self must have no uses in order to lower to Glow.";
    g->block()->eraseInput(0);

    // Create a corresponding runner and store {handle, runner} pair.
    glow::getPyTorchLoaderSettings().preCompilePyTorchModule = true;
    std::unique_ptr<CachingGraphRunner> runner =
        std::make_unique<glow::CachingGraphRunner>(
            g, glow::getHostManager(), getBackendName().c_str(),
            glow::getPyTorchLoaderSettings());

    // Find and parse method_compile_spec
    c10::impl::GenericDict::iterator spec =
        method_compile_spec.find(method.name());
    CHECK(spec != method_compile_spec.end())
        << "Could not find corresponding method_compile_spec for method: "
        << method.name();
    c10::IValue methodSpec = spec->value();
    c10::intrusive_ptr<GlowCompileSpec> gcs;
    try {
      gcs = methodSpec.toCustomClass<GlowCompileSpec>();
    } catch (const std::exception &e) {
      throw std::invalid_argument(
          "method_compile_spec does not match GlowCompileSpec type.");
    }
    std::vector<glow::InputMeta> inputMeta = parseMethodCompileSpec(*gcs);

    // Compile
    auto e = runner->warmCache(inputMeta);
    CHECK(!(bool)e) << ERR_TO_STRING(std::move(e));

    // Bakcend is created on each to_backend call --> use simple consecutive
    // keys for methods.
    handleToRunnerMap_.emplace(key, std::move(runner));
    handles.insert(method.name(), key++);
  }
  return c10::impl::toGenericDict(handles);
}

c10::impl::GenericList
TorchGlowBackend::execute(c10::IValue handle, c10::impl::GenericList inputs) {
  torch::jit::Stack stack;
  for (const auto &i : inputs) {
    torch::jit::push(stack, i);
  }
  auto it = handleToRunnerMap_.find(handle.toInt());
  Error err = glow::ErrorEmpty();
  if (it != handleToRunnerMap_.end()) {
    err = it->second->runOnly(stack);
  } else {
    throw std::out_of_range("Could not find runner for handle " +
                            std::to_string(handle.toInt()));
  }

  if (static_cast<bool>(err)) {
    throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
  }

  c10::List<at::Tensor> output_list;
  while (stack.size() > 0) {
    auto value = torch::jit::pop(stack);
    output_list.emplace_back(value.toTensor());
  }
  return c10::impl::toList(output_list);
}

} // namespace glow
