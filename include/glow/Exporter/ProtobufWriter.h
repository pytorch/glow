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

#ifndef GLOW_EXPORTER_PROTOBUFWRITER_H
#define GLOW_EXPORTER_PROTOBUFWRITER_H

#include "glow/Graph/Graph.h"
#include "glow/Support/Error.h"

#include <fstream>
#include <google/protobuf/text_format.h>

namespace glow {
/// Writes model: graph and weights.
class ProtobufWriter {
protected:
  /// The graph that we are constructing.
  Function *F_;
  /// Output file stream.
  std::ofstream ff_;

  Error writeModel(const ::google::protobuf::Message &modelProto,
                   bool textMode = false);

public:
  /// Constructs new ProtobufWriter object. It will write protopuf messages into
  /// \p modelFilename using graph and constants from \p F.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ProtobufWriter(const std::string &modelFilename, Function *F,
                 Error *errPtr = nullptr, bool writingToFile = true);
};

} // namespace glow

#endif // GLOW_EXPORTER_PROTOBUFWRITER_H
