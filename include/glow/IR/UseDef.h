#ifndef GLOW_IR_USEDEF_H
#define GLOW_IR_USEDEF_H

#include <list>

namespace glow {

/// A UseDef is something that can be an operand for an instruction.
template <typename UserTy, typename UseTy> class UseDef {
public:
  using Use = std::pair<unsigned, UserTy *>;

  virtual ~UseDef() = default;

private:
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
  bool hasUsers() { return users_.size(); }

  /// Returns true if the user \p I is in the list.
  bool hasUser(UserTy *I) {
    for (auto &U : users_) {
      if (U.second == I)
        return true;
    }
    return false;
  }

  /// Replace all of the uses of this value with \p v.
  void replaceAllUsesOfWith(UseTy *v) {
    for (auto &U : users_) {
      U.second->setOperand(U.first, v);
    }
  }

  /// \returns the list of users for this value.
  std::list<Use> &getUsers() { return users_; }
};

} // namespace glow

#endif // GLOW_IR_USEDEF_H
