#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class IRFunction;
class Function;

enum class CompilationMode {
  Train, /// Compile the graph in preperation for training.
  Infer, /// Compile the graph for inference. Notice that this operation
         /// changes the graph in a way that is not reversible.
};

/// Perform optimizations on the IR representation.
void optimize(IRFunction &M, CompilationMode mode);
/// Perform optimizations on the graph representation.
void optimize(Function &G, CompilationMode mode);

/// Lower the high-level neural network operators into low-level lineal algebra
/// operators.
void lower(Function &G, CompilationMode mode);

/// Instrument graph \p G by inserting quantization profile nodes
/// for capturing stats for quantization.
void profileQuantization(Function &G);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
