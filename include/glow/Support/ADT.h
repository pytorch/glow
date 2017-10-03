#ifndef GLOW_SUPPORT_ADT_H
#define GLOW_SUPPORT_ADT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace glow {
using llvm::iterator_range;
using llvm::ArrayRef;
using llvm::MutableArrayRef;
using llvm::StringRef;
} // namespace glow

#endif // GLOW_SUPPORT_ADT_H
