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
#ifndef GLOW_GRAPH_TENSORLAYOUT_H
#define GLOW_GRAPH_TENSORLAYOUT_H

#include <memory>
#include <string>

#include "glow/Graph/Nodes.h"
#include "glow/Support/Error.h"

namespace glow {

/// Layout requirements's Singleton.
template <typename T> class TensorLayoutSingleton {
public:
  /// This is how the verifier, Backend and post-loading canonicalizer can
  /// access layout constraints.
  static T &getInstance() {
    // The Ctor will only be called once.
    static const std::unique_ptr<T> instance{new T{token_{}}};
    return *instance;
  }

protected:
  /// Allow the base class to call any subclass's constructor.
  struct token_ {};

  /// Default Ctor.
  TensorLayoutSingleton() {}

  /// Dtor.
  virtual ~TensorLayoutSingleton() {}

private:
  /// Delete copy constructor.
  TensorLayoutSingleton(const TensorLayoutSingleton &) = delete;

  /// Delete move constructor.
  TensorLayoutSingleton(TensorLayoutSingleton &&) = delete;

  /// Delete copy assignment.
  TensorLayoutSingleton &operator=(const TensorLayoutSingleton &) = delete;

  /// Delete move assignment.
  TensorLayoutSingleton &operator=(TensorLayoutSingleton &&) = delete;
};

/// TensorLayoutDescription - optional helper class for parsing string-based
/// layout.
class TensorLayoutDescription {
  /// Tensor dimensions descriptions for all dimensions.
  std::string dims_[max_tensor_dimensions];
  /// The serialization of the layout.
  std::string serializedLayout_;
  /// Expected number of dimensions.
  size_t numDims_;

public:
  virtual ~TensorLayoutDescription() = default;
  /// Constructs this helper class from a serialized string representation.
  TensorLayoutDescription(const std::string &layoutStr);
  /// Constructs this helper class from an array of strings representing each
  /// individual / pre-separated dimension.
  TensorLayoutDescription(llvm::ArrayRef<std::string> dims);
  /// \returns the alignment of a dimension \p n.
  size_t getAlignment(size_t n) const;
  /// \returns the alignment by parsing dimension string \p s.
  size_t getAlignment(const std::string &s) const;
  /// sets the alignment of dimension \p n to the value \p align. \returns the
  /// new layout serialization for the current dimension.
  llvm::StringRef setAlignment(size_t n, size_t align);
  /// \returns the value of the attribute \p name of a dimension \p n.
  std::string getAttribute(size_t n, llvm::StringRef name) const;
  /// sets the value of attribute \p name to the value \p value. \returns the
  /// new layout serialization for the current dimension.
  llvm::StringRef setAttribute(size_t n, llvm::StringRef name,
                               llvm::StringRef value);
  /// \returns true if both tensor layouts are the same.
  bool isSameLayout(const TensorLayoutDescription &rhs) const;
  /// \returns description of the dimension \p n.
  const llvm::StringRef getNthDimDescription(size_t n) const;
  /// \returns the description of all dimensions.
  llvm::ArrayRef<std::string> getDims() const;
  /// \returns number of dimensions.
  size_t getNumDims() const { return numDims_; }
  /// \returns layout name.
  llvm::StringRef getSerializedLayout() const { return serializedLayout_; }
  /// \returns true if the layout is "*" in all dimensions.
  bool isAnyLayout();
  std::string getDebugDesc() const;

protected:
  /// parse helper: get the custom extensions information. the default, virtual,
  /// implementation just ignores all the data until the end token.
  virtual void parseCustomExtensions(llvm::StringRef &text, unsigned idx);

private:
  /// Constructor helper: Parses the  serialized string.
  void parse(llvm::StringRef text);

  /// parse helper: get the official extensions information.
  void parseOfficialExtensions(llvm::StringRef &text, unsigned idx);

  /// Modifies \p dimStr to remove an extension starting with the prefix \p
  /// name.
  void removeAttribute(const std::string &name, std::string &dimStr);

  /// Rebuilds serializedLayout_ from scratch.
  void reconstructSerialized();
};

/// Interface for finding out layout requirements.
class TensorLayoutCommon {
public:
  /// \return the default n-D layout for Glow.
  virtual std::string getDefaultNDLayout(unsigned dims) const;

  /// \returns layout requirements of the Nth input \p n of a Node \p node.
  virtual std::string getNthInputLayoutRequirements(const Node *node, size_t n);

  /// \returns layout requirements of the Nth result \p n of a Node \p node.
  virtual std::string getNthResultLayoutRequirements(const Node *node,
                                                     size_t n);

  /// \returns true if type \p ty satisfies the \p destLayout layout. If \p
  /// srcLayout is provided, it is taken into account as well.
  virtual bool isSatisfiedBy(TypeRef ty,
                             const TensorLayoutDescription &destLayout,
                             const TensorLayoutDescription *srcLayout) const;

  /// \return layouts for all tensor dimensions.
  virtual llvm::ArrayRef<TensorLayoutDescription> getLayoutsForDims() const;

  /// \returns true if layout equirement verification is enabled.
  bool isEnabled() const { return enabled_; }

protected:
  TensorLayoutCommon();
  TensorLayoutCommon(TensorLayoutCommon &&) = delete;
  TensorLayoutCommon &operator=(const TensorLayoutCommon &) = delete;
  TensorLayoutCommon &operator=(TensorLayoutCommon &&) = delete;
  virtual ~TensorLayoutCommon();

protected:
  bool enabled_;

private:
  std::unordered_map<std::string, TensorLayoutDescription *>
      layoutNameToLayoutDescription_;
};

class CanonicalTensorLayout final
    : public TensorLayoutCommon,
      public TensorLayoutSingleton<CanonicalTensorLayout> {
public:
  CanonicalTensorLayout(token_) {}

  /// \return the default n-D layout for Glow.
  std::string getDefaultNDLayout(unsigned dims) const override;

  /// \returns layout requirements of the Nth input \p n of a Node \p node.
  /// NOTE: Certain nodes are layout agnostic. Others expect their
  /// inputs/outputs to have a canonical format. For some layout agnostic nodes
  /// we need to look at the layout of their inputs to determine the layout of
  /// their outputs, e.g. a batch norm. node, in the canonical representation,
  /// accepts any input layout such as NCHW or NHWC, but, the output is a
  /// propoagation of said layout.
  std::string getNthInputLayoutRequirements(const Node *node,
                                            size_t n) override;

  /// \returns layout requirements of the Nth result \p n of a Node \p node.
  std::string getNthResultLayoutRequirements(const Node *node,
                                             size_t n) override;

  /// \returns true of the node accepts any layout.
  bool acceptsAnyLayout(const Node *node) const;
};

/// Checks if two layout descriptions \p lhs and \p rhs describe the same layout
/// for a value of the type \p ty \returns true if layouts are the same. if \p
/// verbose then print out verbose report.
bool checkSameLayout(llvm::StringRef srcLayoutStr,
                     llvm::StringRef destLayoutStr, TypeRef ty,
                     const Node *parent, const std::string &prefix,
                     const TensorLayoutCommon &TLC, bool verbose = true);

/// Verifies the correctness of tensor layouts in the function \p F using layout
/// requirements interface \p TLC. if \p verbose then print out verbose report.
bool verifyLayouts(const Function &F, TensorLayoutCommon &TLC,
                   bool verbose = true);

} // end namespace glow

#endif // GLOW_GRAPH_TENSORLAYOUT_H
