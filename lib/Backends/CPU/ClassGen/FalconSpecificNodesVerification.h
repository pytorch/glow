#ifdef GLOW_WITH_CPU
// #ifdef GLOW_WITH_FALCON

#include "glow/Graph/VerifierHelper.h"

bool FalconMergedNode::verify() const {
  return true;
}

#endif // GLOW_WITH_FALCON