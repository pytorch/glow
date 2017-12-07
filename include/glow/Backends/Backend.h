#ifndef GLOW_BACKENDS_BACKEND_H
#define GLOW_BACKENDS_BACKEND_H

namespace glow {

class Context;
class Module;
class Value;
class Tensor;
class Variable;

enum class BackendKind {
  Interpreter, // Execute the network with the built-in interpreter.
  OpenCL,
  None,
};

// This is the interface that glow backends need to implement.
class Backend {
public:
  /// Dtor.
  virtual ~Backend() = default;

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
};

/// Create a backend of kind \p kind, to run the module \p M.
Backend *createBackend(BackendKind backendKind, Module *M);

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
