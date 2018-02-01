#ifndef GLOW_OPTIMIZER_OPTIMIZERUTILS_H
#define GLOW_OPTIMIZER_OPTIMIZERUTILS_H

#include <assert.h>
#include <functional>
#include <stdlib.h>
#include <string>

namespace glow {

/// A helper class used to control if a given transformation or
/// optimization may perform transformations and how many of those it is allowed
/// to perform at most.
/// This is useful for debugging, when performing bisections on a specific
/// transformation and trying to figure out which specific change performed by
/// the transformation leads leads to wrong results.
class ChangeManager {
  /// The id of the transformation pass controlled by this object.
  std::string PassId_;
  /// Max number of changes that should be performed by the transformation.
  size_t NumMaxChanges_;
  /// Number of changes performed by the transformation so far.
  size_t NumChanges_;
  /// Number of changes at the time of the last startPass invocation.
  size_t NumChangesSnapshot_;

  void initFromCommandLineOptions();

  /// \returns true if it is OK to perform a transformation.
  bool canChange() const { return NumChanges_ < NumMaxChanges_; }

  /// Notification that the transformation was performed.
  void change() {
    assert(canChange() && "No changes permitted");
    NumChanges_++;
  }

public:
  /// Initialize the change manager.
  /// \param OptId the id of the optimization.
  /// \param Initializer the callback to initialize a change manager, e.g. to
  /// set its NumMaxChanges_. \param MaxNumChanges the default value for the
  /// NumMaxChanges_.

  ChangeManager(std::string PassId, size_t NumMaxChanges = 0xFFFFFF)
      : PassId_(PassId), NumMaxChanges_(NumMaxChanges), NumChanges_(0),
        NumChangesSnapshot_(0) {
    initFromCommandLineOptions();
  }

  /// Ask for permission to change something by means of the transformation.
  /// \returns true if change is permitted, false otherwise.
  bool tryToChange() {
    if (!canChange())
      return false;
    change();
    return true;
  }

  /// \returns true if current invocation of the transformation pass has
  /// tranformed anything.
  bool isChanged() const { return NumChanges_ - NumChangesSnapshot_ > 0; }

  /// Should be invoked at the beginning of every pass run.
  void startPass() { NumChangesSnapshot_ = NumChanges_; }

  void setNumMaxChanges(size_t NumMaxChanges) {
    NumMaxChanges_ = NumMaxChanges;
  }

  const std::string &getOptId() const { return PassId_; }
};

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZERUTILS_H
