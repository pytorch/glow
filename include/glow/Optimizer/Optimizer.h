#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class Module;
class Graph;

enum class CompilationMode {
  Train, // Optimize the module but allow training.
  Infer, // Optimize the module and break training.
};

void optimize(Module &M, CompilationMode mode);
void optimize(Graph &G, CompilationMode mode);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
