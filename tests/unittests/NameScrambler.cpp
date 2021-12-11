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

#include "glow/Support/ZipUtils.h"
#include "onnx/onnx_pb.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <glog/logging.h>

#include <list>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {
llvm::cl::OptionCategory scramblerCat("Scrambler Category");
llvm::cl::opt<std::string>
    inputModelPathOpt("input_model", llvm::cl::desc("Input model zip file"),
                      llvm::cl::Required, llvm::cl::cat(scramblerCat));
llvm::cl::opt<std::string>
    outputModelPathOpt("output_model", llvm::cl::desc("Output model zip file"),
                       llvm::cl::Required, llvm::cl::cat(scramblerCat));
llvm::cl::opt<std::string> inputDeferredWeightsPathOpt(
    "input_deferred_weights",
    llvm::cl::desc("Path to the input deferred weights file"),
    llvm::cl::Optional, llvm::cl::init(""), llvm::cl::cat(scramblerCat));
llvm::cl::opt<std::string> outputDeferredWeightsPathOpt(
    "output_deferred_weights",
    llvm::cl::desc("Path to the output deferred weights file"),
    llvm::cl::Optional, llvm::cl::init(""), llvm::cl::cat(scramblerCat));
llvm::cl::opt<std::string>
    inputPatternOpt("inputs_pattern",
                    llvm::cl::desc("Input file pattern. in_{}.onnx"),
                    llvm::cl::init(""), llvm::cl::cat(scramblerCat));
llvm::cl::opt<std::string>
    outputPatternOpt("outputs_pattern",
                     llvm::cl::desc("Output file pattern. out_{}.onnx"),
                     llvm::cl::init(""), llvm::cl::cat(scramblerCat));
llvm::cl::opt<unsigned> seqStartOpt(
    "seq_start", llvm::cl::desc("Start index of input/output files"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(scramblerCat));
llvm::cl::opt<unsigned> seqLenOpt(
    "seq_len", llvm::cl::desc("Lengths of the input/output file seqquence."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(scramblerCat));
llvm::cl::opt<unsigned>
    methodOpt("method",
              llvm::cl::desc(
                  "Scrambling method: 0: simple tag; 1: pad to the same length;"
                  "2: same lengths, change uppercase letters to a new random "
                  "upper case letter"),
              llvm::cl::Optional, llvm::cl::init(0),
              llvm::cl::cat(scramblerCat));
} // namespace

using namespace glow;

constexpr size_t MAX_PROTO_SIZE = 0x7FFFFFFF;
constexpr bool kCompressed = false;
constexpr int kMaxTrial = 1000;

void scrambleMethod2(std::string &str) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::unordered_set<std::string> used_names;
  std::uniform_int_distribution<> dis(0, 25);
  for (int trial = 0; trial < kMaxTrial; ++trial) {
    std::transform(str.begin(), str.end(), str.begin(), [&](char c) {
      if (c >= 'A' && c <= 'Z') {
        return static_cast<char>('A' + dis(gen));
      } else {
        return c;
      }
    });
    if (used_names.emplace(str).second) {
      return;
    }
  }
  LOG(FATAL) << "Bad luck. Cannot find a unique random name. Try run me again!";
}

std::string makeNewName(const std::string &in) {
  static size_t idx = 0;
  std::stringstream ss;
  if (methodOpt == 2) {
    std::string out = in;
    scrambleMethod2(out);
    return out;
  } else if (methodOpt == 1) {
    ss << idx++;
    std::string tail = ss.str();
    std::string out(in.size() - tail.size(), 'X');
    out += tail;
    if (out.size() != in.size()) {
      LOG(WARNING) << "Cannot pad to the same length of " << in;
    }
    return out;
  } else {
    ss << "X__" << idx++;
    return ss.str();
  }
}

std::string makeNewNodeName(const std::string &in) {
  static size_t idx = 0;
  std::stringstream ss;
  if (methodOpt == 1) {
    ss << idx++;
    std::string tail = ss.str();
    std::string out(in.size() - tail.size(), 'N');
    out += tail;
    if (out.size() != in.size()) {
      LOG(WARNING) << "Cannot pad to the same length of " << in;
    }
    return out;
  } else {
    ss << "N__" << idx++;
    return ss.str();
  }
}

bool parseIO(const std::string &filename, ::ONNX_NAMESPACE::GraphProto &g) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  if (!ff) {
    return false;
  }
  google::protobuf::io::IstreamInputStream fileStream(&ff);
  google::protobuf::io::CodedInputStream codedStream(&fileStream);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE);
#else
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
#endif
  bool yes = g.ParseFromCodedStream(&codedStream);
  if (!yes) {
    return false;
  }
  return true;
}

void rewriteIO(const std::string &filename,
               std::unordered_map<std::string, std::string> &name_map) {
  LOG(INFO) << "Reading file: " << filename;
  ::ONNX_NAMESPACE::GraphProto g;
  if (!parseIO(filename, g)) {
    LOG(ERROR) << "Cannot open " << filename;
    return;
  }
  for (auto &t : *g.mutable_initializer()) {
    const auto &name = t.name();
    if (!name_map.count(name)) {
      LOG(ERROR) << "It's very straight that input " << name
                 << " is not referenced in the net";
      name_map.emplace(name, makeNewName(name));
    }
    t.set_name(name_map.at(name));
  }
  std::string new_filename = filename + ".2";
  LOG(INFO) << "Writing new file: " << new_filename;
  std::ofstream of(new_filename,
                   std::ios::out | std::ios::trunc | std::ios::binary);
  if (!of) {
    LOG(ERROR) << "Cannot open " << new_filename;
    return;
  }
  std::string buffer;
  g.SerializeToString(&buffer);
  of << buffer;
}

std::list<::ONNX_NAMESPACE::TensorProto>
readWeightsAndMaybeCopyData(ZipReader &zip, ZipWriter &zipO, bool compressed) {
  std::list<::ONNX_NAMESPACE::TensorProto> weights;
  auto numWeightsStr = zip.getRecord("weights");
  size_t numWeights = 0;
  numWeights = atoi(numWeightsStr.c_str());
  std::string buffer;
  for (size_t i = 0; i < numWeights; ++i) {
    std::stringstream ss;
    ss << "weight_" << i;
    buffer = zip.getRecord(ss.str());
    weights.emplace_back();
    auto &t = weights.back();
    t.ParseFromString(buffer);

    ss.str("");
    ss << "data_" << i;
    if (zip.hasRecord(ss.str())) {
      buffer = zip.getRecord(ss.str());
      zipO.writeRecord(ss.str(), buffer.c_str(), buffer.size(), compressed);
    }
  }
  return weights;
}

void writeWeights(ZipWriter &zip,
                  const std::list<::ONNX_NAMESPACE::TensorProto> &weights,
                  bool compressed) {
  std::stringstream ss;
  ss << weights.size() << "\n";
  zip.writeRecord("weights", ss.str().c_str(), ss.str().size(), compressed);
  std::string largeBuffer;
  int i = 0;
  // This part is probably quite inefficient as we are deserializing the
  // protobuf to a char buffer and then put it to zip stream. I didn't dig
  // enough to see if we can deserialize it into zip stream directly.
  for (const auto &t : weights) {
    std::stringstream nm;
    nm << "weight_" << i++;
    t.SerializeToString(&largeBuffer);
    zip.writeRecord(nm.str(), largeBuffer.c_str(), largeBuffer.size(),
                    compressed);
  }
}

void scramble() {
  LOG(INFO) << "Input model: " << inputModelPathOpt;
  ::ONNX_NAMESPACE::ModelProto modelDef;
  std::list<::ONNX_NAMESPACE::TensorProto> weights;
  std::unordered_map<std::string, std::string> name_map;
  std::unordered_map<std::string, std::string> node_map;
  {
    LOG(INFO) << "Writing output model to " << outputModelPathOpt;
    std::ofstream ffO(outputModelPathOpt,
                      std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(ffO);
    ZipWriter zipO(&ffO, "test");
    {
      ZipReader zip(inputModelPathOpt);
      std::string buffer;
      buffer = zip.getRecord("model");
      modelDef.ParseFromString(buffer);
      weights = readWeightsAndMaybeCopyData(zip, zipO, kCompressed);
    }

    auto *g = modelDef.mutable_graph();
    for (auto &n : *g->mutable_node()) {
      for (auto &i : *n.mutable_input()) {
        if (!name_map.count(i)) {
          name_map.emplace(i, makeNewName(i));
        }
        i = name_map.at(i);
      }
      for (auto &o : *n.mutable_output()) {
        if (!name_map.count(o)) {
          name_map.emplace(o, makeNewName(o));
        }
        o = name_map.at(o);
      }
      const auto &name = n.name();
      if (!node_map.count(name)) {
        node_map.emplace(name, makeNewNodeName(name));
      }
      n.set_name(node_map.at(name));
    }
    for (auto &i : *g->mutable_input()) {
      const auto &name = i.name();
      if (!name_map.count(name)) {
        name_map.emplace(name, makeNewName(name));
      }
      i.set_name(name_map.at(name));
    }
    for (auto &o : *g->mutable_output()) {
      const auto &name = o.name();
      if (!name_map.count(name)) {
        name_map.emplace(name, makeNewName(name));
      }
      o.set_name(name_map.at(name));
    }
    for (auto &t : weights) {
      const auto &name = t.name();
      if (!name_map.count(name)) {
        LOG(ERROR) << "It's a bit straight that weight " << name
                   << " is not referenced in the net";
        name_map.emplace(name, makeNewName(name));
      }
      t.set_name(name_map.at(name));
    }
    // Look for attributes of a list of strings matching a name and swap for
    // scrambled version. Note that this should be fine because we currently
    // only use a list of strings for vector<NodeValue>.
    for (auto &n : *g->mutable_node()) {
      for (auto &a : *n.mutable_attribute()) {
        if (a.name() == "Predicate") {
          LOG(FATAL) << "Predicate NodeValue unhandled.";
        }
        for (auto &s : *a.mutable_strings()) {
          if (name_map.count(s)) {
            s = name_map[s];
          }
        }
      }
    }

    writeWeights(zipO, weights, kCompressed);
    std::string largeBuffer;
    modelDef.SerializeToString(&largeBuffer);
    zipO.writeRecord("model", largeBuffer.c_str(), largeBuffer.size(),
                     kCompressed);
    zipO.writeEndOfFile();
    ffO.flush();
    ffO.close();
  }

  if (!inputDeferredWeightsPathOpt.empty()) {
    weights.clear();
    // Open the zip writer first in case we need to copy raw tensor data
    // directly from zip reader
    LOG(INFO) << "Writing output deferred weights to "
              << outputDeferredWeightsPathOpt;
    std::ofstream ffO(outputDeferredWeightsPathOpt,
                      std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(ffO);
    ZipWriter zipO(&ffO, "test");

    {
      LOG(INFO) << "Input deferred weights: " << inputDeferredWeightsPathOpt;
      ZipReader zip(inputDeferredWeightsPathOpt);
      weights = readWeightsAndMaybeCopyData(zip, zipO, kCompressed);
      for (auto &t : weights) {
        const auto &name = t.name();
        if (!name_map.count(name)) {
          LOG(ERROR) << "It's very straight that weight " << name
                     << " is not referenced in the net";
          name_map.emplace(name, makeNewName(name));
        }
        t.set_name(name_map.at(name));
      }
    }

    writeWeights(zipO, weights, kCompressed);
    zipO.writeEndOfFile();
    ffO.flush();
    ffO.close();
  }

  size_t input_iter = inputPatternOpt.find("{}");
  CHECK_NE(input_iter, std::string::npos)
      << "Input pattern " << inputPatternOpt << " has to contain {}";
  size_t output_iter = outputPatternOpt.find("{}");
  CHECK_NE(output_iter, std::string::npos)
      << "Output pattern " << outputPatternOpt << " has to contain {}";
  for (unsigned i = seqStartOpt; i < seqLenOpt; ++i) {
    std::string input = inputPatternOpt;
    input.replace(input_iter, 2, std::to_string(seqStartOpt + i));
    rewriteIO(input, name_map);
    std::string output = outputPatternOpt;
    output.replace(output_iter, 2, std::to_string(seqStartOpt + i));
    rewriteIO(output, name_map);
  }
}

void parseCommandLine(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "The name scrambler\n\n"
                                    "Scramble the name for repro files");
}

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);
  scramble();
  return 0;
}
