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

#include "glow/Exporter/ProtobufWriter.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace glow {

ProtobufWriter::ProtobufWriter(const std::string &modelFilename, Function *F,
                               Error *errPtr, bool writingToFile)
    : F_(F) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ProtobufWriter and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    if (writingToFile) {
      // Try to open file for write
      ff_.open(modelFilename,
               std::ios::out | std::ios::trunc | std::ios::binary);
      RETURN_ERR_IF_NOT(ff_,
                        "Can't find the output file name: " + modelFilename,
                        ErrorValue::ErrorCode::MODEL_WRITER_INVALID_FILENAME);
    }
    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

Error ProtobufWriter::writeModel(const ::google::protobuf::Message &modelProto,
                                 bool textMode) {
  {
    ::google::protobuf::io::OstreamOutputStream zeroCopyOutput(&ff_);
    // Write the content.
    if (textMode) {
      RETURN_ERR_IF_NOT(
          google::protobuf::TextFormat::Print(modelProto, &zeroCopyOutput),
          "Can't write to the output file name",
          ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
    } else {
      ::google::protobuf::io::CodedOutputStream codedOutput(&zeroCopyOutput);
      modelProto.SerializeToCodedStream(&codedOutput);
      RETURN_ERR_IF_NOT(
          !codedOutput.HadError(), "Can't write to the output file name",
          ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
    }
  }
  ff_.flush();
  ff_.close();
  return Error::success();
}

} // namespace glow
