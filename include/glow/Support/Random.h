#ifndef GLOW_SUPPORT_RANDOM_H
#define GLOW_SUPPORT_RANDOM_H

namespace glow {

/// \returns the next uniform random number in the range -1..1.
double nextRand();

/// \returns the next uniform random integer in the closed interval [a, b].
int nextRandInt(int a, int b);

} // namespace glow

#endif // GLOW_SUPPORT_RANDOM_H
