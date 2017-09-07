#ifndef GLOW_SUPPORT_ADT_H
#define GLOW_SUPPORT_ADT_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace glow {

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
    if (length_ != RHS.length_) {
      return false;
    }
    return std::equal(begin(), end(), RHS.begin());
  }

  std::vector<T> vec() const { return std::vector<T>(begin(), end()); }
};

/// MutableArrayRef - Represent a mutable reference to an array (0 or more
/// elements consecutively in memory), i.e. a start pointer and a length.  It
/// allows various APIs to take and modify consecutive elements easily and
/// conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the MutableArrayRef. For this reason, it is not in
/// general safe to store a MutableArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
/// Derived from LLVM's MutableArrayRef.
template <typename T> class MutableArrayRef : public ArrayRef<T> {
public:
  using iterator = T *;
  using reverse_iterator = std::reverse_iterator<iterator>;

  MutableArrayRef() = default;

  /// Construct an MutableArrayRef from a single element.
  MutableArrayRef(T &OneElt) : ArrayRef<T>(OneElt) {}

  /// Construct an MutableArrayRef from a pointer and length.
  MutableArrayRef(T *data, size_t length) : ArrayRef<T>(data, length) {}

  /// Construct an MutableArrayRef from a range.
  MutableArrayRef(T *begin, T *end) : ArrayRef<T>(begin, end) {}

  MutableArrayRef(const std::initializer_list<T> &vec) : ArrayRef<T>(vec) {}

  T *data() const { return const_cast<T *>(ArrayRef<T>::data()); }

  iterator begin() const { return data(); }
  iterator end() const { return data() + this->size(); }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// front - Get the first element.
  T &front() const {
    assert(!this->empty());
    return data()[0];
  }

  /// back - Get the last element.
  T &back() const {
    assert(!this->empty());
    return data()[this->size() - 1];
  }

  /// slice(n, m) - Chop off the first N elements of the array, and keep M
  /// elements in the array.
  MutableArrayRef<T> slice(size_t N, size_t M) const {
    assert(N + M <= this->size() && "Invalid specifier");
    return MutableArrayRef<T>(this->data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  MutableArrayRef<T> slice(size_t N) const {
    return slice(N, this->size() - N);
  }

  /// \brief Drop the first \p N elements of the array.
  MutableArrayRef<T> drop_front(size_t N = 1) const {
    assert(this->size() >= N && "Dropping more elements than exist");
    return slice(N, this->size() - N);
  }

  MutableArrayRef<T> drop_back(size_t N = 1) const {
    assert(this->size() >= N && "Dropping more elements than exist");
    return slice(0, this->size() - N);
  }

  T &operator[](size_t Index) const {
    assert(Index < this->size() && "Invalid index!");
    return data()[Index];
  }
};

template <typename T> inline bool operator==(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return LHS.equals(RHS);
}

template <typename T> inline bool operator!=(ArrayRef<T> LHS, ArrayRef<T> RHS) {
  return !(LHS == RHS);
}

template <typename T>
inline bool operator==(MutableArrayRef<T> LHS, MutableArrayRef<T> RHS) {
  return LHS.equals(RHS);
}

template <typename T>
inline bool operator!=(MutableArrayRef<T> LHS, MutableArrayRef<T> RHS) {
  return !(LHS == RHS);
}

} // namespace glow

#endif // GLOW_SUPPORT_ADT_H
