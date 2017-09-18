#ifndef GLOW_SUPPORT_CASTING_H
#define GLOW_SUPPORT_CASTING_H

#include <cassert>
#include <memory>
#include <type_traits>

namespace glow {

template <class TO, class FROM> bool isa(FROM *k) { return (TO::classof(k)); }

template <class TO, class FROM> TO *cast(FROM *k) {
  assert(isa<TO>(k) && "Invalid cast");
  return static_cast<TO *>(k);
}

template <class TO, class FROM> TO *dyn_cast(FROM *k) {
  if (isa<TO>(k))
    return cast<TO>(k);

  return nullptr;
}

} // end namespace glow

#endif // GLOW_SUPPORT_CASTING_H
