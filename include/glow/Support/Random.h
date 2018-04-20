/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_SUPPORT_RANDOM_H
#define GLOW_SUPPORT_RANDOM_H

namespace glow {

/// \returns the next uniform random number in the range -1..1.
double nextRand();

/// \returns the next uniform random integer in the closed interval [a, b].
int nextRandInt(int a, int b);

} // namespace glow

#endif // GLOW_SUPPORT_RANDOM_H
