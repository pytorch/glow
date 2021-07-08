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
#ifndef GLOW_PASSMANAGER_PASSCONFIGTUILS_H
#define GLOW_PASSMANAGER_PASSCONFIGTUILS_H

#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

LLVM_YAML_STRONG_TYPEDEF(unsigned, CompilationModes)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<glow::ConvergenceMode> {
  static void enumeration(IO &io, glow::ConvergenceMode &mode) {
    io.enumCase(mode, "one_pass", glow::ConvergenceMode::OnePass);
    io.enumCase(mode, "until_fixed_point",
                glow::ConvergenceMode::UntilFixedPoint);
  }
};

template <> struct ScalarEnumerationTraits<glow::DCERequiredMode> {
  static void enumeration(IO &io, glow::DCERequiredMode &mode) {
    io.enumCase(mode, "none", glow::DCERequiredMode::None);
    io.enumCase(mode, "before_pass", glow::DCERequiredMode::BeforePass);
  }
};

template <> struct ScalarBitSetTraits<CompilationModes> {
  static void bitset(IO &io, CompilationModes &value) {
    io.bitSetCase(value, "infer",
                  1 << convertEnumToUnsigned(glow::CompilationMode::Infer));
    io.bitSetCase(value, "train",
                  1 << convertEnumToUnsigned(glow::CompilationMode::Train));
  }
};

template <typename Helper> struct SequenceTraits<std::vector<Helper>> {
  static size_t size(IO &io, std::vector<Helper> &configs) {
    return configs.size();
  }

  static Helper &element(IO &io, std::vector<Helper> &configs, size_t index) {
    if (index >= configs.size()) {
      configs.resize(index + 1);
    }
    return configs[index];
  }
};
} // namespace yaml
} // namespace llvm

template <typename T> static T deserializeFromYaml(llvm::StringRef fileName) {
  T result;
  llvm::outs() << "Deserialize from " << fileName << "\n";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);
  assert(!text.getError() && "Unable to open file");

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());
  yin >> result;

  assert(!yin.error() && "Error reading yaml file");

  return result;
}

template <typename T>
static void serializeToYaml(llvm::StringRef fileName, T &value) {
  std::error_code EC;
  llvm::raw_fd_ostream os(fileName, EC);
  CHECK(!EC) << "Could not open output file";
  llvm::yaml::Output yout(os);
  yout << value;
}

#endif // GLOW_PASSMANAGER_PASSCONFIGTUILS_H
