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
#include "GraphScheduler.h"

#include "llvm/Support/CommandLine.h"

using namespace glow;

namespace {
llvm::cl::OptionCategory graphSchedulerCat("Graph Scheduler Options");

llvm::cl::opt<SchedulerKind> graphScheduler(
    llvm::cl::desc("Scheduler to use:"),
    llvm::cl::values(clEnumValN(SchedulerKind::ChildMemSizeBased,
                                "child-mem-size-based",
                                "Use ChildMemSizeBased"),
                     clEnumValN(SchedulerKind::TopologicalSortBased,
                                "topological-sort-based",
                                "Use TopologicalSortBased")),
    llvm::cl::init(SchedulerKind::ChildMemSizeBased),
    llvm::cl::cat(graphSchedulerCat));
} // namespace

namespace glow {
Scheduler *createScheduler(SchedulerKind schedulerKind, Function &G,
                           NodesPtrList &scheduled) {
  switch (schedulerKind) {
  case SchedulerKind::ChildMemSizeBased:
    return new ChildMemSizeBasedScheduler(G, scheduled);
  case SchedulerKind::TopologicalSortBased:
    return new TopologicalSortBasedScheduler(G, scheduled);
  }
  llvm_unreachable("unreachable");
}

void IRFunction::scheduleGraph(NodesPtrList &Schedule) {
  Schedule.clear();
  auto constants = getGraph()->findConstants();
  auto placeholders = getGraph()->findPlaceholders();
  for (auto &N : constants) {
    Schedule.push_back(N);
  }
  for (auto &N : placeholders) {
    Schedule.push_back(N);
  }
  for (auto &N : getGraph()->getMetadataPlaceholders()) {
    Schedule.push_back(N);
  }
  auto numVars = constants.size();
  auto numPlaceholders =
      placeholders.size() + getGraph()->getMetadataPlaceholders().size();
  (void)numVars;
  (void)numPlaceholders;
  std::unique_ptr<Scheduler> scheduler{
      createScheduler(graphScheduler, *getGraph(), Schedule)};
  scheduler->schedule();
  assert(scheduler->getSchedule().size() ==
             getGraph()->getNodes().size() + numPlaceholders + numVars &&
         "All graph nodes have to be scheduled");
}
} // namespace glow
