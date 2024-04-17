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

#ifndef GLOW_BACKENDS_NNPI_NNPIUTILS_H
#define GLOW_BACKENDS_NNPI_NNPIUTILS_H

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

enum NNPIAVXType { NNPI_AVX_NONE = 0, NNPI_AVX_AVX512 };

inline void convertI64toI32(int64_t const *i64Data, int32_t *i32Data,
                            uint32_t elements) {
  for (size_t i = 0; i < elements; i++) {
    i32Data[i] = static_cast<int32_t>(i64Data[i]);
  }
}
void convertI64toI32_AVX512(int64_t const *i64Data, int32_t *i32Data,
                            uint32_t elements);

// Static Dot writer (not thread safe).
class DotWriter {
public:
  static void clear();
  static void addNode(std::string name, std::string label, unsigned color = 0,
                      std::string subGraph = {});
  static void addEdge(std::string src, std::string dst);
  static void writeToFile(std::string filename = {});
  static void addSubGraph(std::string name, std::string label);
  static std::string getHexStr(uint64_t h);

private:
  DotWriter() {} // Should only be used in a static fashion.

  static unsigned graphId_;
  static std::map<std::string, std::set<std::string>> subGraphNodes_;
  static std::map<std::string, std::string> subGraphLabels_;
  static std::set<std::string> edges_;
};

#endif // GLOW_BACKENDS_NNPI_NNPIUTILS_H
