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

#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include <unordered_set>

namespace glow {
using llvm::isa;

// Sorted the std::pair<DAGNode *, uint64_t> based on the second from min to
// max.
bool sortMinMemory(const std::pair<Function *, uint64_t> &a,
                   const std::pair<Function *, uint64_t> &b) {
  return a.second < b.second;
}

void optimizeCommunicationCost(NodeToFunctionMap &partitions,
                               FunctionToNodesMap &nodesSet, Module *mod,
                               uint64_t availableMemory) {
  // Move/Exchange nodes between any two connected partitions, until no gain is
  // get.
  // Step1 Move: Assume Partition1 -> Partition2, try to move nodes from
  // Partition2 to Partition1 if those nodes only use the nodes in
  // Partition1(recursively) and the move won't make Partition1's memory exceeds
  // the memory constraint, and the communication cost is minimized.
  bool gain = true;
  while (gain) {
    // gain is initialized as false, it will be set to be true if there is at
    // least one node can be moved from one set to another set.
    gain = false;
    for (FunctionToNodesMap::iterator it = nodesSet.begin();
         it != nodesSet.end(); ++it) {
      NodesSet &curSet = (*it).second;
      std::vector<Node *> outUsers = getOutUsersWithOnePredecessor(curSet);
      if (outUsers.empty()) {
        continue;
      }
      Function *cur = (*it).first;
      GraphMemInfo curCost = partitions.getGraphMemInfo(cur);
      // Check if a node can be moved to current node set (i.e curSet).
      for (int i = 0, e = outUsers.size(); i < e; i++) {
        // Get the new cost if outUsers[i] is added.
        GraphMemInfo newCurCost =
            updateGraphMemInfoByAddingNode(curSet, curCost, outUsers[i]);

        // Rule 1: this move won't break memory constraint.
        if (newCurCost.getTotalMemSize() > availableMemory) {
          continue;
        }
        // Rule 2: this move won't cause constant duplication.
        bool cont = false;
        for (int j = 0, e1 = outUsers[i]->getNumInputs(); j < e1; j++) {
          auto in = outUsers[i]->getNthInput(j);
          if (isa<Storage>(in.getNode()) && !in.hasOneUse()) {
            cont = true;
            break;
          }
        }
        if (cont) {
          continue;
        }
        // Rule 3: this move won't increase communication cost. Even if this
        // move won't change communication cost, according to rule 1 and rule 2,
        // the memory consumption of the partition where this node (i.e
        // outUsers[i]) belongs can be reduced. Therefore, it may trigger later
        // node movement or partitionsCombine.
        Function *suc = partitions[outUsers[i]];
        uint64_t outMem = getOutMemPerNode(nodesSet[suc], outUsers[i]);
        if (newCurCost.outMemSize - outMem <= curCost.outMemSize) {
          // Move this node to current node set.
          curSet.insert(outUsers[i]);
          nodesSet[suc].erase(outUsers[i]);
          curCost = newCurCost;
          // Update the partitions.
          partitions.add(outUsers[i], cur);
          partitions.setGraphMemInfo(cur, newCurCost);
          if (nodesSet[suc].empty()) {
            // It is possible that after moving a node from Partition2 to
            // Partition1, Partition2 become empty. Remove the empty partition.
            partitions.deletePartition(suc);
            nodesSet.erase(suc);
            mod->eraseFunction(suc);
          } else {
            GraphMemInfo newCost = getGraphMemInfo(nodesSet[suc]);
            partitions.setGraphMemInfo(suc, newCost);
          }
          gain = true;
        }
      }
    }
  }
}

void partitionsCombine(NodeToFunctionMap &partitions,
                       FunctionToNodesMap &nodesSet, Module *mod,
                       uint64_t availableMemory) {

  size_t origPartitions = 0;

  // Do the combination until the size of partitions is stable.
  while (partitions.getPartitions().size() != origPartitions) {
    origPartitions = partitions.getPartitions().size();
    // Rule 1:
    for (FunctionToNodesMap::iterator it = nodesSet.begin();
         it != nodesSet.end(); ++it) {
      std::vector<Node *> outUsers = getOutUsers((*it).second);
      if (outUsers.empty()) {
        continue;
      }

      bool flag = true;
      for (int i = 1, e = outUsers.size(); i < e; i++) {
        if (partitions[outUsers[i]] != partitions[outUsers[i - 1]]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        // This partition only has one successor.
        Function *cur = (*it).first;
        Function *suc = partitions[outUsers[0]];
        NodesSet tmp = (nodesSet.find(suc))->second;
        GraphMemInfo cost1 = partitions.getGraphMemInfo(cur);
        GraphMemInfo cost2 = partitions.getGraphMemInfo(suc);
        if (cost1.getTotalMemSize() + cost2.getTotalMemSize() -
                cost1.outMemSize <
            availableMemory) {
          // We can combine the two partitions to fit one device.
          for (NodesSet::iterator it2 = tmp.begin(); it2 != tmp.end(); ++it2) {
            partitions.add(*it2, cur);
          }
          GraphMemInfo newCost;
          newCost.constMemSize = cost1.constMemSize + cost2.constMemSize;
          newCost.inMemSize =
              cost1.inMemSize + cost2.inMemSize - cost1.outMemSize;
          newCost.outMemSize = cost2.outMemSize;
          partitions.setGraphMemInfo((*it).first, newCost);
          (*it).second.insert(tmp.begin(), tmp.end());
          partitions.deletePartition(suc);
          nodesSet.erase(suc);
          mod->eraseFunction(suc);
        }
      }
    }
  }
}

DeviceIDTy
assignLogicalDeviceID(NodeToFunctionMap &mapping,
                      const std::map<std::string, BackendInfo> &backendMap) {
  DeviceIDTy logicalDeviceID = 0;

  std::map<std::string, std::vector<Function *>> backendFuncMap;
  for (auto &func : mapping.getPartitions()) {
    // Traverse the partitions, and get list of partitions with each
    // backendName.
    auto backendName = mapping.getPartitionBackendName(func);
    if (backendFuncMap.find(backendName) == backendFuncMap.end()) {
      backendFuncMap.emplace(backendName, std::vector<Function *>{func});
    } else {
      backendFuncMap[backendName].push_back(func);
    }
  }

  // For each type of the backend, assign the logicalDevice ID.
  for (const auto &p : backendFuncMap) {
    if (mapping.getPartitions().size() <= backendMap.at(p.first).num) {
      // There is enough device with this backendName, no need to adjust the
      // logical ID.
      for (auto &func : p.second) {
        mapping.appendLogicalDeviceID(func, logicalDeviceID++);
      }
      continue;
    }
    // Get the list of functions with current BackendName, and sort it based on
    // used memory from min to max.
    std::vector<std::pair<Function *, uint64_t>> nodeSize;
    for (size_t i = 0, e = p.second.size(); i < e; i++) {
      Function *function = p.second[i];
      uint64_t totalMem = mapping.getGraphMemInfo(function).getTotalMemSize();
      nodeSize.push_back(std::make_pair(p.second[i], totalMem));
    }
    std::sort(nodeSize.begin(), nodeSize.end(), sortMinMemory);

    // Assume we have n devices(NOTE: here the n devices have the same available
    // memory size, and the following algorithm can find the accurate result. If
    // the memory size are differnt, this assignment issue will be a NP problem
    // -- multiple knapsack problem, and the following algorithm becomes greedy
    // and the result may not be optimal), and m partitions, where m > n. If
    // these m partitions can be assigned to n devices, there must be 1 device
    // have at least (m - 1)/n + 1 partitions(Pigeonhole principle). Based on
    // this theory, the algorithm is: Given N devices, and M partitions:
    // Step 1 : sort the partitions from min to max based on their memory usage.
    std::sort(nodeSize.begin(), nodeSize.end(), sortMinMemory);
    // Step 2 : let n = N, m = M.
    size_t m = p.second.size();
    size_t n = backendMap.at(p.first).num;
    while (m > 0) {
      // Step 3 : find the first k partitions whose total memory usage still
      // under the memory limitation (k should be max).
      uint64_t usedMem = 0;
      size_t numOfPartitionsWithSameID = (m - 1) / n + 1;
      size_t start = p.second.size() - m;
      size_t i;
      for (i = start; i < p.second.size(); i++) {
        if (usedMem + nodeSize[i].second > backendMap.at(p.first).memSize) {
          break;
        }
        usedMem += nodeSize[i].second;
      }
      // Step 4 : if k = start - i found in step 3 is smaller than (m - 1) / n +
      // 1, this means we can't find a proper assignment to fit the number of
      // devices. Assign each partition with a unique logicalDevice ID and
      // return.
      if (i - start < numOfPartitionsWithSameID) {
        // Can't find a proper assignment. Assign each partition a unique
        // logicalDevice ID and return;
        logicalDeviceID = 0;
        for (auto &func : mapping.getPartitions()) {
          mapping.appendLogicalDeviceID(func, logicalDeviceID++);
        }
        return logicalDeviceID;
      }

      // Step 5 : Assign these partitions which are assigned to one device with
      // the same logical ID.
      for (size_t j = start; j < i; j++) {
        mapping.appendLogicalDeviceID(nodeSize[j].first, logicalDeviceID);
      }
      logicalDeviceID++;
      // Step 6 : Update the left number of devices and partitions.
      n--;
      m = m - (i - start);
    }
  }
  return logicalDeviceID;
}
} // namespace glow
