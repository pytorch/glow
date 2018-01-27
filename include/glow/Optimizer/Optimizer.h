#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class Module;
class Graph;

enum class CompilationMode {
  TrainDebug, /// Compile the graph for training and add extra instrumentation
              /// that enable unit testing and debugging, e.g., saving gradient
              /// results.
  Train,      /// Compile the graph in preperation for training.
  Infer,      /// Compile the graph for inference. Notice that this operation
              /// changes the graph in a way that is not reversible.
};

/// Optimize the module.
/// \returns true if anything was changed by the optimizer.
bool optimize(Module &M, CompilationMode mode);
/// Optimize the graph.
/// \returns true if anything was changed by the optimizer.
bool optimize(Graph &G, CompilationMode mode);

/// Lower the high-level neural network operators into low-level lineal algebra
/// operators.
void lower(Graph &G, CompilationMode mode);

/// Instrument graph \p G by inserting quantization profile nodes
/// for capturing stats for quantization.
void profileQuantization(Graph &G);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
