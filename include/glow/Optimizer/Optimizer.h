#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class Module;
class Graph;

enum class CompilationMode {
  TrainDebug, /// Compiler the graph for training and add extra instrumentation
              /// that enable unit testing and debugging.
  Train,      /// Compile the graph in preperation for training.
  Infer,      /// Compiler the graph for inference. Notice that this operation
              /// changes the graph in a way that is not reversible.
};

void optimize(Module &M, CompilationMode mode);
void optimize(Graph &G, CompilationMode mode);

/// Lower the high-level neural network operators into low-level lineal algebra
/// operators.
void lower(Graph &G, CompilationMode mode);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
