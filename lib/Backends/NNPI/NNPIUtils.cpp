/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "NNPIUtils.h"
#include "DebugMacros.h"
#include <fstream>
#include <sstream>

unsigned DotWriter::graphId_(0);
std::map<std::string, std::set<std::string>> DotWriter::subGraphNodes_;
std::map<std::string, std::string> DotWriter::subGraphLabels_;
std::set<std::string> DotWriter::edges_;

static const std::string &getColorString(unsigned i) {
  // Taking colors from the SVG scheme
  static const std::vector<std::string> nodeColors = {
      "mistyrose",  // function
      "lightgreen", // host resource
      "lightblue",  // normal device resource
      "plum",       // static device resource
      "lightcoral", // p2p device resource
      "wheat",      // drt device resource
      "lightgray",  // reserved
      "sandybrown", // reserved
      "turquoise",  // reserved
      "seagreen",   // reserved
  };
  return nodeColors.at(i % nodeColors.size());
}

void DotWriter::clear() {
  DotWriter::subGraphNodes_.clear();
  DotWriter::subGraphLabels_.clear();
  DotWriter::edges_.clear();
}

void DotWriter::addNode(std::string name, std::string label, unsigned color,
                        std::string subGraph) {
  std::ostringstream os;
  os << name << " [\n";
  os << "\tlabel = \"" << label << "\"\n";
  os << "\tstyle=\"filled,rounded\"\n";
  os << "\tfillcolor=" << getColorString(color) << "\n";
  os << "];\n";
  if (!subGraph.empty()) {
    subGraphNodes_[subGraph].insert(os.str() /*name*/);
  }
}

void DotWriter::addEdge(std::string src, std::string dst) {
  edges_.insert(src + " -> " + dst + ";\n");
}

void DotWriter::writeToFile(std::string filename) {
  if (filename.empty()) {
    filename = "dot_graph";
  }
  filename = filename + std::to_string(graphId_++) + ".dot";
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    LOG(INFO) << "Failed to write dor file: " << filename;
    return;
  }
  outFile << "digraph {\n";
  outFile << "\tedge[color = black];\n";
  outFile << "\trank = TB;\n";
  outFile << "\tnode[shape = Mrecord, penwidth=2];\n";
  outFile << "\n";

  for (const auto &sg : subGraphNodes_) {
    outFile << "subgraph " << "cluster_" << sg.first << " {\n";
    outFile << "\tlabel = \"" << subGraphLabels_.at(sg.first) << "\";\n";
    for (const auto &n : sg.second) {
      outFile << n; //<< ";\n";
    }
    outFile << "}\n";
  }
  for (const auto &e : edges_) {
    outFile << e;
  }

  outFile << "\n";
  outFile << "\t}\n";
}

void DotWriter::addSubGraph(std::string name, std::string label) {
  subGraphLabels_[name] = label;
}

std::string DotWriter::getHexStr(uint64_t h) {
  std::ostringstream os;
  os << std::hex << h;
  return os.str();
}
