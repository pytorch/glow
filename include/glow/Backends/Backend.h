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
  OpenCL,      // Run the code on an OpenCL device.
  JIT,         // Compile and run the code on the host.
  None,
};

// This is the interface that glow backends need to implement.
class Backend {
public:
  /// Dtor.
  virtual ~Backend() = default;

  /// Wipe out the state of the backend.
  virtual void clear() = 0;

  /// Prepare the interpreter for execution of new code.
  virtual void init() = 0;

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  virtual void doForwardPass(bool isTrain) = 0;
};

/// Create a backend of kind \p kind, to run the module \p M.
Backend *createBackend(BackendKind backendKind, Module *M);

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
