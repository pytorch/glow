#ifndef NOETHER_ADT_H
#define NOETHER_ADT_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace noether {

/// ArrayRef - represent a constant reference to an array.
/// Derived from LLVM's ArrayRef.
template <typename T> class ArrayRef {
public:
  using iterator = const T *;
  using const_iterator = const T *;
  using size_type = size_t;
  using reverse_iterator = std::reverse_iterator<iterator>;

private:
  const T *data_ = nullptr;
  size_type length_ = 0;

public:
  /// Construct an empty ArrayRef.
  ArrayRef() = default;

  /// Construct an ArrayRef from a single element.
  ArrayRef(const T &one) : data_(&one), length_(1) {}

  /// Construct an ArrayRef from a pointer and length.
  ArrayRef(const T *data, size_t length) : data_(data), length_(length) {}

  /// Construct an ArrayRef from a range.
  ArrayRef(const T *begin, const T *end) : data_(begin), length_(end - begin) {}

  /// Construct an ArrayRef from a std::initializer_list.
  ArrayRef(const std::initializer_list<T> &vec)
      : data_(vec.begin() == vec.end() ? (T *)nullptr : vec.begin()),
        length_(vec.size()) {}

  /// Construct an ArrayRef from a std::vector.
  template <typename A>
  ArrayRef(const std::vector<T, A> &vec)
      : data_(vec.data()), length_(vec.size()) {}

  iterator begin() const { return data_; }
  iterator end() const { return data_ + length_; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  bool empty() const { return length_ == 0; }

  const T *data() const { return data_; }

  size_t size() const { return length_; }

  /// Drop the first element of the array.
  ArrayRef<T> drop_front() const {
    assert(size() >= 1 && "Array is empty");
    return ArrayRef(data() + 1, size() - 1);
  }

  /// Drop the last element of the array.
  ArrayRef<T> drop_back(size_t N = 1) const {
    assert(size() >= 1 && "Array is empty");
    return ArrayRef(data(), size() - 1);
  }

  const T &operator[](size_t Index) const {
    assert(Index < length_ && "Invalid index!");
    return data_[Index];
  }

  /// equals - Check for element-wise equality.
  bool equals(ArrayRef RHS) const {
    if (length_ != RHS.length_)
      return false;
    return std::equal(begin(), end(), RHS.begin());
  }

  std::vector<T> vec() const { return std::vector<T>(begin(), end()); }
};

template <typename T> inline bool operator==(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return LHS.equals(RHS);
}

template <typename T> inline bool operator!=(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return !(LHS == RHS);
}
} // namespace noether

#endif // NOETHER_ADT_H
