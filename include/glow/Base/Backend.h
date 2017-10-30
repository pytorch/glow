#ifndef GLOW_BASE_BACKEND_H
#define GLOW_BASE_BACKEND_H

#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

namespace glow {

class Context;
class Module;
class Value;
class Tensor;
class Variable;

// This is the interface that glow backends need to implement.
class Backend {
public:
  /// Dtor.
  virtual ~Backend() {}

  /// Wipe out the state of the interpreter.
  virtual void clear() = 0;

  /// Prepare the interpreter for execution of new code.
  virtual void init() = 0;

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  virtual void doForwardPass(bool isTrain) = 0;

  /// Registers the external tensor \p t, that's owned by the graph, as mapped
  /// to the value \p v.
  virtual void registerGraphTensor(const Value *v, Tensor *t) = 0;

  /// \returns a pointer to the tensor that is saved under \p v.
  virtual Tensor *getTensor(const Variable *v) const = 0;

  /// \returns a pointer to the gradient tensor that matches \p v. Notice
  /// that this API is only valid when the module is compiled in training mode.
  virtual Tensor *getGradTensor(const Variable *v) const = 0;
};

} // namespace glow

#endif // GLOW_BASE_BACKEND_H
