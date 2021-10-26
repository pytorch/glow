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
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Flags/Flags.h"
#include "glow/Partitioner/PartitionerTypes.h"
#include "glow/Support/Support.h"

#include <unordered_set>

using llvm::isa;

namespace glow {

namespace {
/// Used to sort 2 Nodes by their name, i.e. n1->name < n2->name order.
auto compFunc = [](const Node *n1, Node *n2) -> bool {
  return n1->compareByName(*n2);
};
constexpr uint32_t MB = 1024 * 1024;
} // namespace

/// The nodes in function \p F which be grouped into levels based on how far
/// (the longest distance) they are from the roots.
BFSLevel getBFSLevel(Function *F) {
  // The current set of nodes needs to be visited
  std::unordered_set<Node *> cur;
  // A map between a node and its level.
  llvm::DenseMap<Node *, int> nodeLevel;

  // Get the roots set (i.e. the nodes without users).
  for (auto &node : F->getNodes()) {
    if (node.getNumUsers() == 0) {
      // A root has no users.
      cur.insert(&node);
      nodeLevel[&node] = 0;
    }
  }

  // Create the node to level map by traversing the nodes with BFS order.
  BFSLevel bfs;
  int level = 0;
  int current = 0;
  bfs.push_back(std::vector<Node *>());
  level++;
  while (current < level) {
    std::unordered_set<Node *> nodes;
    for (std::unordered_set<Node *>::iterator it = cur.begin(); it != cur.end();
         ++it) {
      Node *N = *it;
      for (size_t j = 0, e = N->getNumInputs(); j < e; ++j) {
        Node *in = N->getNthInput(j).getNode();
        if (isa<Storage>(in)) {
          continue;
        }
        nodes.insert(in);
        nodeLevel[in] = level;
      }
    }
    if (nodes.size() > 0) {
      bfs.push_back(std::vector<Node *>());
      level++;
      cur = std::move(nodes);
    }
    current++;
  }

  // Based on the node to level map, group these nodes by levels.
  for (llvm::DenseMap<Node *, int>::iterator it = nodeLevel.begin();
       it != nodeLevel.end(); ++it) {
    Node *in = (*it).first;
    int level = (*it).second;
    bfs[level].push_back(in);
  }

  // Sort the nodes of each level by name to make sure the nodes sequence are
  // the same for different run.
  for (int i = 0; i < level; i++) {
    std::sort(bfs[i].begin(), bfs[i].end(), compFunc);
  }
  return bfs;
}

/// Given \p nodes, return a list of nodes who are not in this set but use any
/// node in this set.
std::vector<Node *> getOutUsers(const NodesSet &nodes) {
  NodesSet used;
  for (NodesSet::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    for (auto &U : cur->getUsers()) {
      if (nodes.count(U.getUser())) {
        continue;
      }
      used.insert(U.getUser());
    }
  }

  std::vector<Node *> ret(used.begin(), used.end());
  std::sort(ret.begin(), ret.end(), compFunc);
  return ret;
}

/// Given \p nodes, return a list of nodes who are not in this set but use only
/// the nodes in this set or constant.
std::vector<Node *> getOutUsersWithOnePredecessor(const NodesSet &nodes) {
  NodesSet used;
  for (NodesSet::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    for (auto &U : cur->getUsers()) {
      Node *user = U.getUser();
      if (nodes.count(user)) {
        continue;
      }
      bool flag = true;
      for (size_t i = 0, e = user->getNumInputs(); i < e; i++) {
        Node *in = user->getNthInput(i).getNode();
        if (llvm::isa<Storage>(in) || nodes.count(in)) {
          continue;
        }
        flag = false;
        break;
      }
      if (flag) {
        used.insert(user);
      }
    }
  }

  std::vector<Node *> ret(used.begin(), used.end());
  std::sort(ret.begin(), ret.end(), compFunc);
  return ret;
}

/// \returns the memory usage of the output caused by \p node who has users not
/// in the set \p nodes.
uint64_t getOutMemPerNode(const NodesSet &nodes, const Node *node) {
  uint64_t ret = 0;
  for (size_t i = 0, e = node->getNumResults(); i < e; i++) {
    NodeValue nodeVal = node->getNthResult(i);
    for (auto &U : nodeVal.getUsers()) {
      Node *user = U.getUser();
      if (nodes.find(const_cast<Node *>(user)) == nodes.end()) {
        ret += node->getType(i)->getSizeInBytes();
        break;
      }
    }
  }
  return ret;
}

/// Given a node, \return the NodeSet of all nodes that create the results
/// for any of the inputs of this node (i.e. input of inputs)
NodesSet getInputs(const Node *node) {
  NodesSet result;
  for (size_t i = 0, e = node->getNumInputs(); i < e; i++) {
    Node *input = node->getNthInput(i).getNode();
    Storage *in = llvm::dyn_cast<Storage>(input);
    if (!in) {
      result.insert(input);
    }
  }
  return result;
}

uint64_t getNodeMemUsage(const Node *node) {
  if (node->getKind() == Kinded::Kind::SaveNodeKind) {
    return 0;
  }
  uint64_t size = 0;
  for (size_t i = 0, e = node->getNumInputs(); i < e; i++) {
    Storage *in = llvm::dyn_cast<Storage>(node->getNthInput(i).getNode());
    if (in) {
      auto ty = in->getType();
      size += ty->getSizeInBytes();
    }
  }
  return size;
}

float getNodeComputeTime(const Node *node, const BackendInfo &backendInfo) {
  // This code assumes all ops are BW limited from SRAM; except
  // if the input does not fit in SRAM -- then it is DRAM BW limited
  float peakDramBw = backendInfo.peakDramBw;
  float peakSramBw = backendInfo.peakSramBw;
  uint64_t sramCapacity = backendInfo.sramCapacity;
  float peakCompute = backendInfo.peakCompute;

  // compute memory side bytes for inputs from DRAM, SRAM.
  // TODO: think about whether this is better off computed inside a Node.

  int n = node->getNumInputs();
  uint64_t sizeDram = 0;
  uint64_t sizeSram = 0;
  if (node->getKind() == Kinded::Kind::SaveNodeKind) {
    return 0.0f;
  }
  // The memory bytes for embedding table lookups is data dependent,
  // so it needs to be calculated as per the number of indices accessed.
  if (node->getKind() == Kinded::Kind::SparseLengthsWeightedSumNodeKind) {
    auto *SLWSN = llvm::dyn_cast<SparseLengthsWeightedSumNode>(node);
    // compute how many entries of the embedding table we look up
    auto numLookups = SLWSN->getIndices().dims().front();
    // compute how many bytes we read per lookup
    auto tableSize = SLWSN->getData().getType()->getSizeInBytes();
    auto numRows = SLWSN->getData().dims().front();
    auto sizePerLookup = tableSize / numRows;
    // compute total bytes read
    uint64_t sizeInput = numLookups * sizePerLookup;

    // tables are usually large and fit in DRAM
    sizeDram += sizeInput;
    // we also read the indices, weights and lengths arrays
    sizeSram += SLWSN->getIndices().getType()->getSizeInBytes();
    sizeSram += SLWSN->getWeights().getType()->getSizeInBytes();
    sizeSram += SLWSN->getLengths().getType()->getSizeInBytes();
  } else if (node->getKind() == Kinded::Kind::SparseLengthsSumNodeKind) {
    auto *SLSN = llvm::dyn_cast<SparseLengthsSumNode>(node);
    // compute how many entries of the embedding table we look up
    auto numLookups = SLSN->getIndices().dims().front();
    // compute how many bytes we read per lookup
    auto tableSize = SLSN->getData().getType()->getSizeInBytes();
    auto numRows = SLSN->getData().dims().front();
    auto sizePerLookup = tableSize / numRows;
    // compute total bytes read
    uint64_t sizeInput = numLookups * sizePerLookup;

    // tables are usually large and fit in DRAM
    sizeDram += sizeInput;
    // we also read the indices and lengths arrays
    sizeSram += SLSN->getIndices().getType()->getSizeInBytes();
    sizeSram += SLSN->getLengths().getType()->getSizeInBytes();
  } else if (node->getKind() ==
             Kinded::Kind::
                 FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind) {
    auto *FRQSLWSN =
        llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(node);
    // compute how many entries of the embedding table we look up
    auto numLookups = FRQSLWSN->getIndices().dims().front();
    // compute how many bytes we read per lookup
    auto tableSize = FRQSLWSN->getData().getType()->getSizeInBytes();
    auto numRows = FRQSLWSN->getData().dims().front();
    auto sizePerLookup = tableSize / numRows;
    // compute total bytes read
    uint64_t sizeInput = numLookups * sizePerLookup;

    // tables are usually large and fit in DRAM
    sizeDram += sizeInput;

    // we also read the indices, weights and lengths arrays
    sizeSram += FRQSLWSN->getIndices().getType()->getSizeInBytes();
    sizeSram += FRQSLWSN->getWeights().getType()->getSizeInBytes();
    sizeSram += FRQSLWSN->getLengths().getType()->getSizeInBytes();
  } else if (node->getKind() ==
             Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind) {
    auto *FRQSLSN =
        llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(node);
    // compute how many entries of the embedding table we look up
    auto numLookups = FRQSLSN->getIndices().dims().front();
    // compute how many bytes we read per lookup
    auto tableSize = FRQSLSN->getData().getType()->getSizeInBytes();
    auto numRows = FRQSLSN->getData().dims().front();
    auto sizePerLookup = tableSize / numRows;
    // compute total bytes read
    uint64_t sizeInput = numLookups * sizePerLookup;

    // tables are usually large and fit in DRAM
    sizeDram += sizeInput;

    // we also read the indices and lengths arrays
    sizeSram += FRQSLSN->getIndices().getType()->getSizeInBytes();
    sizeSram += FRQSLSN->getLengths().getType()->getSizeInBytes();
  } else {
    // for all other ops, iterate through all inputs and get size in bytes
    for (int i = 0; i < n; i++) {
      auto ty = node->getNthInput(i).getType();
      uint64_t sizeInput = ty->getSizeInBytes();
      if (sizeInput > sramCapacity) {
        sizeDram += sizeInput;
      } else {
        sizeSram += sizeInput;
      }
    }
  }

  // Repeat for outputs
  for (size_t i = 0, e = node->getNumResults(); i < e; i++) {
    auto myty = node->getType(i);
    uint64_t sizeOutput = myty->getSizeInBytes();
    if (sizeOutput > sramCapacity) {
      sizeDram += sizeOutput;
    } else {
      sizeSram += sizeOutput;
    }
  }

  // Calculate compute ops. Currently only computed for Matmul, Conv, FC
  // TODO: think about whether this is better off computed inside a Node.
  uint64_t totalOps = 0;
  switch (node->getKind()) {
  case Kinded::Kind::MatMulNodeKind: {
    auto *MMN = llvm::dyn_cast<MatMulNode>(node);
    auto lhsDims = MMN->getLHS().dims();
    auto rhsDims = MMN->getRHS().dims();
    totalOps = 2 * lhsDims[0] * lhsDims[1] * rhsDims[1];
    break;
  }
  case Kinded::Kind::FullyConnectedNodeKind: {
    auto *FCN = llvm::dyn_cast<FullyConnectedNode>(node);
    auto inputDims = FCN->getInput().dims();
    auto wtDims = FCN->getWeights().dims();
    totalOps = 2 * inputDims[0] * inputDims[1] * wtDims[0];
    break;
  }
#ifdef GLOW_WITH_HABANA
  case Kinded::Kind::HabanaFullyConnectedNodeKind: {
    auto *FCN = llvm::dyn_cast<HabanaFullyConnectedNode>(node);
    auto inputDims = FCN->getInput().dims();
    auto wtDims = FCN->getWeights().dims();
    totalOps = 2 * inputDims[0] * inputDims[1] * wtDims[0];
    break;
  }
#endif
  case Kinded::Kind::ConvolutionNodeKind: {
    auto *CN = llvm::dyn_cast<ConvolutionNode>(node);
    auto resultDims = CN->getResult().dims();
    // Get the product of batch, output height, output dims, output channels
    totalOps = resultDims[0];
    for (size_t i = 1, e = resultDims.size(); i < e; i++) {
      totalOps *= resultDims[i];
    }
    // Multiply in kernel height, kernel width
    auto kernelDims = CN->getKernels();
    totalOps *= kernelDims[0] * kernelDims[1];
    // Multiply in input channels/groups
    auto inputChannels = CN->getInput().dims()[1];
    auto nGroups = CN->getGroup();
    totalOps *= (inputChannels * 1.0 / nGroups);
    break;
  }
#ifdef GLOW_WITH_HABANA
  case Kinded::Kind::HabanaConvolutionNodeKind: {
    auto *CN = llvm::dyn_cast<HabanaConvolutionNode>(node);
    auto resultDims = CN->getResult().dims();
    // Get the product of batch, output height, output dims, output channels
    totalOps = resultDims[0];
    for (size_t i = 1, e = resultDims.size(); i < e; i++) {
      totalOps *= resultDims[i];
    }
    // Multiply in kernel height, kernel width
    auto kernelDims = CN->getKernels();
    totalOps *= kernelDims[0] * kernelDims[1];
    // Multiply in input channels/groups
    auto inputChannels = CN->getInput().dims()[1];
    auto nGroups = CN->getGroup();
    totalOps *= (inputChannels * 1.0 / nGroups);
    break;
  }
#endif
  default:
    break;
  }

  // Compute compute roofline as max of flops, DRAM, SRAM BW
  // See https://bit.ly/2UdJ3mz
  // Add epsilons to prevent seg faults on uninitialized peak values.
  return std::max(totalOps * 1.0f / std::max(peakCompute, 1e-6f),
                  std::max(sizeDram * 1.0f / std::max(peakDramBw, 1e-6f),
                           sizeSram * 1.0f / std::max(peakSramBw, 1e-6f)));
}

/// Given nodes set \p currNodes and its memory usage info \p info, \returns the
/// new memory usage if \p newNode is added into \p currNodes.
GraphMemInfo updateGraphMemInfoByAddingNode(const NodesSet &currNodes,
                                            const GraphMemInfo &info,
                                            Node *newNode) {
  GraphMemInfo ret = info;

  // Collect the used NodeValues (Storage nodes and outputs from the nodes
  // outside of currNodes).
  std::set<NodeValue> usedNodeValue;
  for (auto N : currNodes) {
    for (size_t i = 0, e = N->getNumInputs(); i < e; i++) {
      NodeValue nodeVal = N->getNthInput(i);
      if (currNodes.count(nodeVal.getNode()) == 0) {
        usedNodeValue.insert(nodeVal);
      }
    }
  }
  // Calculate new outMemSize.
  NodesSet newNodes = currNodes;
  newNodes.insert(newNode);
  uint64_t newSize = 0;
  for (auto *node : newNodes) {
    if (auto *SN = llvm::dyn_cast<SaveNode>(node)) {
      // SaveNode is a special case since it has no users but always writes out.
      newSize += SN->getOutput().getType()->getSizeInBytes();
    } else {
      newSize += getOutMemPerNode(newNodes, node);
    }
  }
  ret.outMemSize = newSize;

  // The memory usage changes due to newNode's inputs:
  for (size_t i = 0, e = newNode->getNumInputs(); i < e; i++) {
    if (llvm::isa<SaveNode>(newNode) && i == SaveNode::OutputIdx) {
      continue;
    }
    NodeValue nodeVal = newNode->getNthInput(i);
    Node *N = nodeVal.getNode();

    if (usedNodeValue.count(nodeVal)) {
      // This input has been considered already, nothing to do.
      continue;
    }

    Storage *in = llvm::dyn_cast<Storage>(N);
    if (in) {
      // Node uses placeholders or constants which are not used in this set
      // before, need to add the memory.
      uint64_t size = in->getType()->getSizeInBytes();
      if (in->getKind() == Kinded::Kind::ConstantKind) {
        ret.constMemSize += size;
      } else {
        Placeholder *ph = llvm::dyn_cast<Placeholder>(N);
        // If PH is static treat like a constant.
        if (ph->isStatic()) {
          ret.constMemSize += size;
          ret.deferredConstMemSize += size;
        } else {
          // PlaceHolder for Input.
          ret.inMemSize += size;
          ret.inputCount += 1;
        }
      }
      usedNodeValue.insert(nodeVal);
      continue;
    }

    if (!currNodes.count(N)) {
      // In this case, this input is not a storage type node nor belongs
      // to this subgraph. Therefore, when creating paritions, we need to add
      // a PlaceHolder for the data from outside.
      ret.inMemSize += nodeVal.getType()->getSizeInBytes();
      ret.inputCount += 1;
      usedNodeValue.insert(nodeVal);
    }
  }

  for (size_t i = 0, e = newNode->getNumResults(); i < e; i++) {
    auto nodeVal = newNode->getNthResult(i);
    for (auto &U : nodeVal.getUsers()) {
      if (currNodes.count(U.getUser()) == 0) {
        // The nodeVal (i.e. the ith output of newNode) is not used in
        // currNodes:
        continue;
      }
      // Assume newNode -> node1, where node1 belongs to currNodes set. Before
      // newNode is added, node1's input size (from newNode) should be added
      // into inMemSize. But afater newNode is added, the input size should be
      // removed.
      ret.inMemSize -= nodeVal.getType()->getSizeInBytes();
      ret.inputCount -= 1;
      break;
    }
  }

  return ret;
}

GraphMemInfo getGraphMemInfo(const NodesSet &nodes, unsigned contextCount) {
  GraphMemInfo ret;
  ret.contextCount = contextCount;
  NodesSet nodeSet;
  for (NodesSet::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    ret = updateGraphMemInfoByAddingNode(nodeSet, ret, cur);
    nodeSet.insert(cur);
  }
  return ret;
}

GraphMemInfo getFunctionMemory(Function *func) {
  GraphMemInfo graphMem;

  for (auto cons : func->findConstants()) {
    graphMem.constMemSize += cons->getType()->getSizeInBytes();
  }

  // Gather all other functions in the module for peer resource usage counting.
  std::vector<const Function *> otherFuns;
  std::copy_if(func->getParent()->getFunctions().begin(),
               func->getParent()->getFunctions().end(),
               std::back_inserter(otherFuns),
               [func](Function *F) { return func != F; });

  // Walk thru all Placeholders in the function to accumulate input and
  // output mem size. These utility functions check the users of the PH to
  // determine if the PH is an input or an output.
  for (auto &place : func->findPlaceholders()) {
    if (place->isStatic()) {
      graphMem.constMemSize += place->getType()->getSizeInBytes();
      graphMem.deferredConstMemSize += place->getType()->getSizeInBytes();
    } else {
      if (isInput(place, *func)) {
        graphMem.inMemSize += place->getType()->getSizeInBytes();
        graphMem.inputCount += 1;
        // Check if this placeholder is the output of a peer function.
        if (isOutput(place, otherFuns)) {
          graphMem.inputFromPeerCount += 1;
        }
      }
      if (isOutput(place, *func)) {
        graphMem.outMemSize += place->getType()->getSizeInBytes();
      }
    }
  }

  return graphMem;
}

std::set<Kinded::Kind> generateNodeKindsSet(llvm::StringRef names) {
  std::set<Kinded::Kind> nodeKindsSet;
  llvm::StringRef::size_type pos = names.find(',');
  while (pos != llvm::StringRef::npos) {
    nodeKindsSet.insert(getKindFromNodeName(names.substr(0, pos)));
    names = names.substr(pos + 1);
    pos = names.find(',');
  }
  if (!names.empty()) {
    nodeKindsSet.insert(getKindFromNodeName(names));
  }
  return nodeKindsSet;
}

void logPartitionInfo(const NodeToFunctionMap &partitions) {
  int i = 0;
  for (Function *subF : partitions.getPartitions()) {
    LOG(INFO) << "\t Partition " << i++ << ":\n"
              << "\t\t Name :\t" << subF->getName().str() << "\n"
              << "\t\t BackendKind :\t"
              << partitions.getPartitionBackendName(subF) << "\n"
              << "\t\t context count :\t"
              << partitions.getGraphMemInfo(subF).contextCount << "\n"
              << "\t\t total Memory :\t"
              << partitions.getGraphMemInfo(subF).getTotalMemSize() << "\n"
              << "\t\t\t input size:\t"
              << partitions.getGraphMemInfo(subF).inMemSize << "\n"
              << "\t\t\t input count :\t"
              << partitions.getGraphMemInfo(subF).inputCount << "\n"
              << "\t\t\t input only from peers count :\t"
              << partitions.getGraphMemInfo(subF).inputFromPeerCount << "\n"
              << "\t\t\t output size:\t"
              << partitions.getGraphMemInfo(subF).outMemSize << "\n"
              << "\t\t\t constant size:\t"
              << partitions.getGraphMemInfo(subF).constMemSize << "\n"
              << "\t\t\t\t non-deferred constant size:\t"
              << partitions.getGraphMemInfo(subF).constMemSize -
                     partitions.getGraphMemInfo(subF).deferredConstMemSize
              << "\n"
              << "\t\t\t\t deferred constant size:\t"
              << partitions.getGraphMemInfo(subF).deferredConstMemSize << "\n";
    // This may be called before logicalDevices are assigned so check before
    // printing.
    if (partitions.getLogicalDeviceIDList(subF).size()) {
      LOG(INFO) << "\t\t LogicalDeviceIDs :\t"
                << partitions.getLogicalDeviceIDList(subF)[0] << "\n";
    }
  }
}

void printSlsTableInfo(std::vector<SLSTableInfo>::iterator start,
                       std::vector<SLSTableInfo>::iterator end,
                       bool verbose_only) {
  if (start >= end) {
    return;
  }
  std::stringstream ss;
  ss << "(numBytesInTable(MB), deviceID, cost, cost/numBytesInTable) "
     << strFormat(" - %zu tables -", end - start) << "\n";
  while (start < end) {
    const auto tableSizeInMB = (float)start->numBytesInTable / MB;
    const auto costPerByte = tableSizeInMB == 0
                                 ? "nan"
                                 : std::to_string(start->cost / tableSizeInMB);
    ss << "        " << tableSizeInMB << "        " << start->deviceId
       << "      " << start->cost << "      " << costPerByte << std::endl;
    start++;
  }
  if (verbose_only) {
    VLOG(1) << ss.str();
  } else {
    LOG(INFO) << ss.str();
  }
}

void printSlsTableInfo(std::vector<SLSTableInfo> &slsTables,
                       bool verbose_only) {
  printSlsTableInfo(slsTables.begin(), slsTables.end(), verbose_only);
}

void printSlsDeviceInfo(const std::vector<SLSDeviceInfo> &slsDevices,
                        const std::vector<NodesSet> &nodesets,
                        const unsigned contextCount, bool verbose_only) {
  std::stringstream ss;
  ss << "(deviceId, used_memory(MB), free_memory(MB), cost, "
        "node_size, cost/used_memory)"
     << strFormat(" - %zu devices -", slsDevices.size()) << "\n";
  for (const auto &d : slsDevices) {
    const auto deviceId = d.deviceId;
    const auto meminfo = getGraphMemInfo(nodesets[deviceId], contextCount);
    const auto usedMem = (float)meminfo.getTotalMemSize() / MB;
    const auto availMem = (float)d.memAvailableInBytes / MB;
    const auto freeMem = availMem - usedMem;
    const auto costPerUsedMemory =
        usedMem == 0 ? "nan" : std::to_string(d.currentCost / usedMem);
    ss << "    " << deviceId << "         " << usedMem << "         " << freeMem
       << "      " << d.currentCost << "      " << nodesets[deviceId].size()
       << "      " << costPerUsedMemory << "\n";
  }
  if (verbose_only) {
    VLOG(1) << ss.str();
  } else {
    LOG(INFO) << ss.str();
  }
}

bool isSLSNode(const Node *node) {
  return (
      node->getKind() ==
          glow::Kinded::Kind::
              FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind ||
      node->getKind() ==
          glow::Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind ||
      node->getKind() == glow::Kinded::Kind::
                             RowwiseQuantizedSparseLengthsWeightedSumNodeKind ||
      node->getKind() == glow::Kinded::Kind::SparseLengthsSumNodeKind ||
      node->getKind() == glow::Kinded::Kind::SparseLengthsWeightedSumNodeKind ||
      node->getKind() == glow::Kinded::Kind::EmbeddingBagNodeKind ||
      node->getKind() ==
          glow::Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind);
}

bool checkNodeInputsAllKind(const Node *node, glow::Kinded::Kind kind) {
  bool allSameKind = true;
  for (auto i = 0; i < node->getNumInputs(); i++) {
    auto nodeInput = node->getNthInput(i);
    allSameKind &= nodeInput.getNode()->getKind() == kind;
  }
  return allSameKind;
}

Error assignSlsTableToFirstAvailableDevice(
    SLSTableInfo &table, std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<NodesSet> &nodesets,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount,
    std::unordered_map<Node *, size_t> &addedSLSNodes) {
  DCHECK(slsDevices.size() == nodesets.size() &&
         slsDevices.size() == frontierValues.size());
  auto addedNodeDeviceId = addedSLSNodes.find(table.node);
  if (addedNodeDeviceId != addedSLSNodes.end()) {
    table.deviceId = addedNodeDeviceId->second;
    return Error::success();
  }

  bool deviceFound = false;
  for (auto &d : slsDevices) {
    const auto deviceId = d.deviceId;
    // Calculate the memory needed if we merge SLS and its neighboring nodes
    // into existing partition
    auto nodesSetd = nodesets[deviceId];
    nodesSetd.insert(table.node);
    nodesSetd.insert(table.neighbors.begin(), table.neighbors.end());
    auto meminfo = getGraphMemInfo(nodesSetd, contextCount);
    const auto totalSize = meminfo.getTotalMemSize();
    if (d.memAvailableInBytes >= totalSize) {
      d.currentCost += (size_t)table.cost;
      table.deviceId = deviceId;
      frontierValues[deviceId].insert(table.frontier.begin(),
                                      table.frontier.end());
      for (auto &nb : table.neighbors) {
        if (isSLSNode(nb)) {
          addedSLSNodes.insert({nb, deviceId});
        }
      }
      nodesets[deviceId].swap(nodesSetd);
      deviceFound = true;
      break;
    }
  }
  if (!deviceFound) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    "SLS Balancing Partitioning Error: Not enough memory");
  }
  return Error::success();
}

Error assignSlsTablesToDevices(
    std::vector<SLSTableInfo> &slsTables,
    std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount) {
  if (slsTables.empty()) {
    LOG(INFO) << "SLS tables empty!";
    return Error::success();
  }
  // Keep a copy of input parameters, so that ScopeGuard could restore
  // inputs in case of error.
  std::vector<SLSTableInfo> slsTablesCopy = slsTables;
  std::vector<SLSDeviceInfo> slsDevicesCopy = slsDevices;
  std::vector<std::unordered_set<NodeValue>> frontierValuesCopy =
      frontierValues;
  ScopeGuard restoreInputsOnError([&]() {
    slsTables.swap(slsTablesCopy);
    slsDevices.swap(slsDevicesCopy);
    frontierValues.swap(frontierValuesCopy);
  });

  // Now sort SLS tables by size decreasing
  VLOG(1) << "SLS tables sorted by size decreasing";
  std::sort(slsTables.begin(), slsTables.end(),
            [](const SLSTableInfo &l, const SLSTableInfo &r) {
              return l.numBytesInTable > r.numBytesInTable;
            });

  // slsTables is in sorted order decreasingly by numBytesInTable.
  // The tables between [slsTablesLeft, slsTableRight) are large tables that
  // have numBytesInTable > BigTableThresholdBytes.
  // slsTablesLeft and slsTablesRight will be both pointed to slsTables.begin()
  // if we could not find any large tables.
  auto slsTablesLeft = slsTables.begin();
  auto slsTableRight = slsTables.end();
  if (slsTablesLeft->numBytesInTable >
      glow::runtime::flags::BigTableThresholdBytes) {
    for (auto it = slsTables.begin(); it < slsTables.end(); it++) {
      if (it->numBytesInTable <= glow::runtime::flags::BigTableThresholdBytes) {
        slsTableRight = it;
        break;
      }
    }
  } else {
    // No large table found.
    slsTablesLeft = slsTables.begin();
    slsTableRight = slsTables.begin();
  }

  // We first assign large tables to devices. After allocation, each device
  // should has roughly the same size.
  LOG(INFO) << strFormat("Now assign %zu large tables to %zu devices.",
                         (slsTableRight - slsTablesLeft), slsDevices.size());
  // Print Large SLS tables
  VLOG(1) << "Large tables by size decreasing: ";
  printSlsTableInfo(slsTablesLeft, slsTableRight);
  std::vector<NodesSet> nodesets(slsDevices.size());
  std::unordered_map<Node *, size_t> addedSLSNodes;
  while (slsTablesLeft < slsTableRight) {
    // Sort devices by size increasingly.
    std::sort(slsDevices.begin(), slsDevices.end(),
              [&nodesets, contextCount](const SLSDeviceInfo &l,
                                        const SLSDeviceInfo &r) {
                auto lTotalSize =
                    getGraphMemInfo(nodesets[l.deviceId], contextCount)
                        .getTotalMemSize();
                auto rTotalSize =
                    getGraphMemInfo(nodesets[r.deviceId], contextCount)
                        .getTotalMemSize();
                return lTotalSize < rTotalSize;
              });
    VLOG(1) << "Devices sorted by used memory increasing: ";
    printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                       true /* verbose_only */);

    // Pick the first that fits
    auto &table = *slsTablesLeft;
    RETURN_IF_ERR(assignSlsTableToFirstAvailableDevice(
        table, slsDevices, nodesets, frontierValues, contextCount,
        addedSLSNodes));
    slsTablesLeft++;
  }
  VLOG(1) << "Done assigning large tables, devices info: ";
  printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                     true /* verbose_only */);

  // Now let us assign small size tables.
  // First sort tables by cost decreasingly. For each table, we would like to
  // assign it to the device with lowest cost.
  LOG(INFO) << strFormat("Now assign %zu small tables to %zu devices.",
                         (slsTables.end() - slsTablesLeft), slsDevices.size());
  if (slsTablesLeft < slsTables.end()) {
    std::sort(slsTablesLeft, slsTables.end(),
              [](const SLSTableInfo &l, const SLSTableInfo &r) {
                return l.cost > r.cost;
              });
  }
  VLOG(1) << "Small tables by cost decreasingly: ";
  printSlsTableInfo(slsTablesLeft, slsTables.end());

  while (slsTablesLeft < slsTables.end()) {
    // Sort devices by cost increasingly.
    std::sort(slsDevices.begin(), slsDevices.end(),
              [](const SLSDeviceInfo &l, const SLSDeviceInfo &r) {
                return l.currentCost < r.currentCost;
              });

    VLOG(1) << "Devices sorted by cost increasing: ";
    printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                       true /* verbose_only */);

    // Pick the first that fits
    auto &table = *slsTablesLeft;
    RETURN_IF_ERR(assignSlsTableToFirstAvailableDevice(
        table, slsDevices, nodesets, frontierValues, contextCount,
        addedSLSNodes));
    slsTablesLeft++;
  }
  // Print final device info
  LOG(INFO) << "Done assigning small tables, final devices info: ";
  printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                     false /* verbose_only */);
  restoreInputsOnError.dismiss();
  return Error::success();
}

Error assignSlsTablesToDevicesGreedy(
    std::vector<SLSTableInfo> &slsTables,
    std::vector<SLSDeviceInfo> &slsDevices,
    std::vector<std::unordered_set<NodeValue>> &frontierValues,
    const unsigned contextCount) {
  if (slsTables.empty()) {
    LOG(INFO) << "SLS tables empty!";
    return Error::success();
  }
  // Keep a copy of input parameters, so that ScopeGuard could restore
  // inputs in case of error.
  std::vector<SLSTableInfo> slsTablesCopy = slsTables;
  std::vector<SLSDeviceInfo> slsDevicesCopy = slsDevices;
  std::vector<std::unordered_set<NodeValue>> frontierValuesCopy =
      frontierValues;
  ScopeGuard restoreInputsOnError([&]() {
    slsTables.swap(slsTablesCopy);
    slsDevices.swap(slsDevicesCopy);
    frontierValues.swap(frontierValuesCopy);
  });

  // Now sort SLS tables by size decreasing
  VLOG(1) << "SLS tables sorted by size decreasing" << std::endl;
  std::sort(slsTables.begin(), slsTables.end(),
            [](const SLSTableInfo &l, const SLSTableInfo &r) {
              return l.numBytesInTable > r.numBytesInTable;
            });

  // Print SLS tables
  printSlsTableInfo(slsTables);

  // Now assign SLS Nodes to devices
  std::vector<NodesSet> nodesets(slsDevices.size());
  std::unordered_map<Node *, size_t> addedSLSNodes;
  for (auto &table : slsTables) {

    // Sort by cost increasing
    std::sort(slsDevices.begin(), slsDevices.end(),
              [](const SLSDeviceInfo &l, const SLSDeviceInfo &r) {
                return l.currentCost < r.currentCost;
              });

    VLOG(1) << "Devices sorted by cost increasing" << std::endl;
    printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                       true /* verbose_only */);

    // Pick the first that fits
    RETURN_IF_ERR(assignSlsTableToFirstAvailableDevice(
        table, slsDevices, nodesets, frontierValues, contextCount,
        addedSLSNodes));
  }
  // Print final device info
  LOG(INFO) << "Devices sorted by cost increasing: ";
  printSlsDeviceInfo(slsDevices, nodesets, contextCount,
                     false /* verbose_only */);
  restoreInputsOnError.dismiss();
  return Error::success();
}

} // namespace glow
