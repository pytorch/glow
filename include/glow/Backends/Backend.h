#ifndef GLOW_BACKENDS_BACKEND_H
#define GLOW_BACKENDS_BACKEND_H

#include <llvm/ADT/StringRef.h>

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;
class Function;

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

  /// Save the bundle for a later standalone execution.
  virtual void save(llvm::StringRef outputDir);

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  virtual void doForwardPass(bool isTrain) = 0;

  /// @name Backend transform methods for different phases.
  /// These methods are called by the compiler before code generation and gives
  /// the backend an opportunity to transform the graph before IRGen. The
  /// backend may insert target specific nodes. The backend is responsible for
  /// cleaning up after itself.
  /// \returns True if the graph was modified.
  ///@{
  virtual bool transformPreLowering(Function *F) { return false; }
  virtual bool transformPostLowering(Function *F) { return false; }
  /// @}
};

/// Create a backend of kind \p kind, to run the IR function \p M.
Backend *createBackend(BackendKind backendKind, IRFunction *M);

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
