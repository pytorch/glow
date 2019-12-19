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
/// \file This file describes the high-level API for converting a function
/// from one type to another.
#ifndef GLOW_CONVERTER_FUNCTIONCONVERTER_H
#define GLOW_CONVERTER_FUNCTIONCONVERTER_H

#include "glow/Base/Type.h"

#include <utility> // For std::pair.

namespace glow {
class PlaceholderBindings;
class Function;
class Node;
struct NodeValue;
class Placeholder;
class Tensor;

/// This class implements the high-level APIs used to convert a function
/// from one type to another. The actual conversions must be implemented
/// by derived classes.
class FunctionConverter {
protected:
  /// The function to be converted.
  Function &function_;

  /// \return the type that \p out needs to have at the end of the conversion
  /// procedure. In other words, this is the type this value will have at the
  /// end of ::convert.
  /// E.g., let say we want to convert:
  /// \verbatim
  /// res = matmul float
  /// \endverbatim
  /// into
  /// \verbatim
  /// res = matmul fp16
  /// \endverbatim
  /// The target type for res is fp16.
  ///
  /// Using this information, the conversion procedure will insert a conversion
  /// of \p out from this type to the current type of \p out.
  /// \verbatim
  /// res = matmul fp16
  /// ... = convert fp16 res to res's current type
  /// \endverbatim
  ///
  /// If nullptr is returned or the returned type is identical to the current
  /// type of the related value, no conversion will be inserted by the
  /// conversion procedure.
  virtual TypeRef getTargetTypeForOutput(const NodeValue &out) const;

  /// \return the type that the input operand described by \p idx-th input of
  /// \p use needs to have at the end of the conversion procedure. In other
  /// words, this is the type this value will have at the end of ::convert.
  /// E.g., let say we want to convert:
  /// \verbatim
  /// res = matmul float A, B
  /// \endverbatim
  /// into
  /// \verbatim
  /// res = matmul fp16 A, B
  /// \endverbatim
  /// The target type for A (i.e., (matmul, 0)) is fp16.
  ///
  /// Using this information, the conversion procedure will insert a conversion
  /// of (\p node, \p idx) from its current type to the returned type.
  /// \verbatim
  /// convertedA = convert A's current type A to returned type
  /// res = matmul fp16 convertedA, B
  /// \endverbatim
  ///
  /// If nullptr is returned or the returned type is identical to the current
  /// type of the related value, no conversion will be inserted by the
  /// conversion procedure.
  virtual TypeRef getTargetTypeForInput(const Node &use, unsigned idx) const;

  /// Check if \p node can be converted.
  /// \return false if \p node shouldn't be considered for conversion.
  virtual bool canConvert(const Node &node) const;

  /// Create a conversion with \p val as input and \p destTy as the destination
  /// type in \p function, given \p node. In other words, creates something like
  /// cast val to destTy. \p isInput represents if this is converting an input.
  virtual Node *createConversion(Function &function, const Node &node,
                                 NodeValue &val, TypeRef destTy,
                                 bool isInput) = 0;

  /// Given a \p conversion, get its output value.
  /// The default implementation returns the zero-th result.
  /// If a conversion node defined more than one value, this
  /// method must be overloaded.
  virtual NodeValue getConversionOutput(Node &conversion) const;

  /// Mutate the outputs of \p node to the expected output target
  /// type (\see getTargetTypeForOutput) and insert the conversions
  /// to preserve the type consistency with the rest of the network.
  void convertOutputs(Node &node);

  /// Insert conversion node for each input of \p node that don't
  /// match getTargetTypeForInput.
  void convertInputs(Node &node);

  /// Convert the \p input tensor to the \p destTy destination type.
  virtual void convertTensor(Tensor &input, TypeRef destTy) = 0;

  /// Morph \p node into its final form. For the most part
  /// this method should be a noop and just return \p node.
  /// However, this hook provides a way to perform changes
  /// on more than just the type of the inputs and outputs,
  /// like changing the opcode of an operation.
  ///
  /// \warning \p node must not be deleted.
  ///
  /// \pre All the inputs of \p node have been converted to
  ///      their target type using ::getTargetTypeForInput.
  /// \pre All the results of \p node have been converted to
  ///      their target type using ::getTargetTypeForOutput.
  ///
  /// \return the final morphed node.
  virtual Node &morphNode(Node &node);

  /// Hook to perform some post processing on the final morphed node.
  virtual void postProcessing(Node &node);

  /// Hook to do a final clean-up after all operations have been converted.
  virtual void cleanUp() {}

public:
  /// Create a function converter for \p F.
  ///
  /// \note This method will modify \p F when calling ::convert.
  ///       If one wants to keep the original function around,
  ///       they need to clone it before creating this converter.
  FunctionConverter(Function &F) : function_(F) {}

  virtual ~FunctionConverter() {}

  /// Convert \p F according to ::getTargetTypeForOutput and
  /// ::getTargetTypeForInput.
  ///
  /// The high level algorithm looks like:
  /// \code
  /// for each node in function:
  ///   insert conversions for the inputs of node
  ///   update the inputs of node to use the results of the conversions
  ///   mutate the type of the outputs of node
  ///   insert conversions for the outputs of node
  ///   morph node
  ///   postProcessing node
  /// cleanUp
  /// \endcode
  void convert();

  /// Modify the type of \p placeholder according to getTargetTypeForOutput.
  /// If the \p context is provided and \p placeholder has a backing tensor,
  /// this tensor is also updated.
  /// Note: If \p placeholder is used in functions other than F, changes to
  /// those functions will be made as well to accommodate the converted
  /// placeholder.
  void convertPlaceholder(Placeholder &placeholder,
                          PlaceholderBindings *context);
};
} // namespace glow
#endif
