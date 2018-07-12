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

#include "MultiCPUBackend.h"
#include "MultiCPUFunction.h"

#include "glow/Graph/FunctionGraph.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Utils.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/raw_ostream.h"

using namespace glow;

namespace glow {
Backend *createMultiCPUBackend() { return new MultiCPUBackend(); }
} // namespace glow

std::unique_ptr<CompiledFunction>
MultiCPUBackend::compile(std::vector<std::unique_ptr<IRFunction>> IR,
                         FunctionGraph &G) const {
  std::vector<std::unique_ptr<CompiledFunction>> CFs;
  for (auto &F : IR) {
    CFs.emplace_back(CPUBackend::compile(std::move(F)));
  }
  return llvm::make_unique<MultiCPUFunction>(std::move(G), std::move(CFs));
}

namespace {
static Node *singleInput(Node *node) {
  Node *nonVarInput = nullptr;

  for (unsigned i = 0, e = node->getNumInputs(); i < e; i++) {
    Node *in = node->getNthInput(i).getNode();
    if (!llvm::isa<Variable>(in)) {
      if (nonVarInput)
        return nullptr;
      nonVarInput = in;
    }
  }
  return nonVarInput;
}
} // end namespace

FunctionGraph MultiCPUBackend::partition(Function *F) const {
  FunctionList tasks;
  std::unordered_map<Node *, Function *> taskMap;

  // Assign nodes to partitioned functions.
  GraphPostOrderVisitor visitor(*F);
  for (auto *node : visitor.getPostOrder()) {
    if (llvm::isa<Variable>(node))
      continue;

    auto *in = singleInput(node);
    if (in && in->getNumUsers() == 1) {
      auto ti = taskMap.find(in);
      assert(ti != taskMap.end());
      taskMap.emplace(node, ti->second);
    } else {
      auto *newF = F->getParent()->createFunction(
          std::string(F->getName()) + "_part" + std::to_string(tasks.size()));
      tasks.emplace_back(newF);
      taskMap.emplace(node, newF);
    }
  }

  // Create Variables and corresponding Save nodes along function edges.
  auto *mod = F->getParent();
  std::unordered_set<Variable *> channelVars;
  for (auto *node : visitor.getPostOrder()) {
    if (llvm::isa<Variable>(node))
      continue;

    auto it = taskMap.find(node);
    assert(it != taskMap.end() && "Node isn't part of any partition");
    auto *fn = it->second;
    for (unsigned inp = 0, e = node->getNumInputs(); inp < e; inp++) {
      auto const &input = node->getNthInput(inp);
      if (llvm::isa<Variable>(input.getNode()))
        continue;

      auto it = taskMap.find(input.getNode());
      assert(it != taskMap.end() && "Node isn't part of any partition");
      auto *inputF = it->second;
      if (fn == inputF)
        continue;

      auto *tmp = mod->createVariable(
          input.getType(), std::string(input->getName()) + "_tmp",
          VisibilityKind::Private, Variable::TrainKind::None);
      channelVars.emplace(tmp);

      Node *inNode = input.getNode();
      unsigned inResNo = input.getResNo();
      inNode->getNthResult(inResNo).replaceAllUsesOfWith(tmp);
      auto *save = F->createSave("tmp", inNode->getNthResult(inResNo), tmp);
      taskMap.emplace(save, inputF);
    }
  }

  // Clone nodes into their target tasks, and link cloned inputs.
  std::unordered_map<Node *, Node *> nodeMap;
  GraphPostOrderVisitor visitor2(*F);
  for (auto *node : visitor2.getPostOrder()) {
    if (llvm::isa<Variable>(node))
      continue;

    auto *clone = node->clone();
    nodeMap.emplace(node, clone);
    auto fit = taskMap.find(node);
    assert(fit != taskMap.end() && "Node isn't part of any partition");
    fit->second->addNode(clone);
    for (unsigned inp = 0, e = clone->getNumInputs(); inp < e; inp++) {
      auto const &input = clone->getNthInput(inp);
      auto it = nodeMap.find(input.getNode());
      if (it == nodeMap.end()) {
        assert(llvm::isa<Variable>(input.getNode()) &&
               "Could not find a node mapping");
        continue;
      }
      clone->setNthInput(input.getResNo(), it->second);
    }
  }

  return FunctionGraph(tasks, channelVars);
}
