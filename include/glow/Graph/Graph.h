#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"

#include <list>
#include <unordered_map>

namespace glow {

class Function;

using TypesList = std::list<Type>;
using VariablesList = std::list<Variable *>;

/// Represents the compute graph.
class Graph final : public Named {
private:
  /// A uniqued list of types in the module. Types in this list can be equated
  /// by comparing their addresses.
  TypesList types_{};
  /// A list of variables that the graph owns.
  VariablesList vars_;

  /// Unique index for producing unique names.
  size_t uniqueIdx_;

  /// \returns unique name with a prefix \p Name.
  std::string uniqueName(llvm::StringRef Name);

  /// Functions, that belong to the module.
  std::unordered_map<std::string, Function *> funcs_;

public:
  Graph(llvm::StringRef Name = {}) : Named(Name), uniqueIdx_(1) {}

  ~Graph();

  /// Assign unique name for node \p N.
  void uniquifyName(Node *N);

  /// Inserts the variable \p V to the list of variables.
  Variable *addVar(Variable *V) {
    assert(state_ < State::IRGenerated &&
           "Trying to add Variable when IR is already generated.");
    uniquifyName(V);
    vars_.push_back(V);
    return V;
  }

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(const Type &T);

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims);

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims, float scale,
                     int32_t offset);

  /// Return a pointer to a uniqued type \p t in the current module.
  /// The new type is identical to \p T, with a new shape \p dims.
  TypeRef uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<size_t> dims);

  /// Return the void type.
  TypeRef getVoidTy();

  /// @name High-level variables builder.
  /// @{

  Variable *createVariable(
      TypeRef T, llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);

  Variable *createVariable(
      ElemKind T, llvm::ArrayRef<size_t> dims, llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);

  Variable *createVariable(
      ElemKind T, llvm::ArrayRef<size_t> dims, float scale, int32_t offset,
      llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);

  /// @}

  /// Erase a variable from the graph.
  void eraseVariable(Variable *N);
  void eraseVariable(VariablesList::iterator I);

  /// \returns a pointer to the first variable with the name \p name or nullptr
  /// if no node has this name.
  Variable *getVariableByName(llvm::StringRef name);

  /// \returns the list of variables that the graph owns.
  VariablesList &getVars() { return vars_; }

  const VariablesList &getVars() const { return vars_; }

  void storeFunction(llvm::StringRef name, Function *F) {
    funcs_[name] = F;
  }

  Function *getFunction(llvm::StringRef name) {
    return funcs_[name];
  }
};

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
