#ifndef GLOW_IR_USEDEF_H
#define GLOW_IR_USEDEF_H

#include "llvm/ADT/SmallVector.h"

#include <list>

namespace glow {

/// A UseDef is something that can be an operand for an instruction.
template <typename UserTy, typename UseTy, typename Use> class UseDef {
  /// A list of users. Notice that the same user may appear twice in the list.
  /// This is typically a very short list.
  std::list<Use> users_{};

public:
  UseDef() = default;

  /// Removes the use \p U from the uselist.
  void removeUse(Use U) {
    auto it = std::find(users_.begin(), users_.end(), U);
    assert(it != users_.end() && "User not in list");
    users_.erase(it);
  }
  /// Adds the use \p U.
  void addUse(Use U) { users_.push_back(U); }

  /// \returns True if the value has some users.
  bool hasUsers() const { return !users_.empty(); }

  /// \returns true if there is a single use to this value.
  bool hasOneUse() const { return users_.size() == 1; }

  /// \returns the number of users that the value has.
  unsigned getNumUsers() const { return users_.size(); }

  /// Returns true if the user \p I is in the list.
  bool hasUser(const UserTy *I) const {
    for (const auto &U : users_) {
      if (U.get() == I) {
        return true;
      }
    }
    return false;
  }

  /// \returns the list of users for this value.
  std::list<Use> &getUsers() { return users_; }

  /// \returns the list of users for this value.
  const std::list<Use> &getUsers() const { return users_; }
};

} // namespace glow

#endif // GLOW_IR_USEDEF_H
