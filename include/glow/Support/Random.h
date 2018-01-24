#ifndef GLOW_SUPPORT_RANDOM_H
#define GLOW_SUPPORT_RANDOM_H

namespace glow {

/// \returns the next uniform random number in the range -1..1.
double nextRand();

/// \returns the next uniform random integer that is either 0 or 1.
int nextRandInt01();

} // namespace glow

#endif // GLOW_SUPPORT_RANDOM_H
