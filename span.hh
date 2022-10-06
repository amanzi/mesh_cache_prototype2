#pragma once 


#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>


/// A workalike for std::span from C++20 (only dynamic-extent, without ranges
/// support).
template<class T>
struct span {
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  using iterator = pointer; // implementation-defined
  using reverse_iterator = std::reverse_iterator<iterator>;

  constexpr span() noexcept : span(nullptr, nullptr) {}
  constexpr span(pointer p, size_type sz) : span(p, p + sz) {}
  constexpr span(pointer p, pointer q) : p(p), q(q) {}
  template<std::size_t N>
  constexpr span(element_type (&a)[N]) : span(a, N) {}
  /// \warning Destroying \a C leaves this object dangling if it owns its
  ///   elements.  This implementation does not check for "borrowing".
  template<class C,
    class = std::enable_if_t<std::is_convertible_v<
      std::remove_pointer_t<decltype(void(std::size(std::declval<C &&>())),
        std::data(std::declval<C &&>()))> (*)[],
      T (*)[]>>>
  constexpr span(C && c) : span(std::data(c), std::size(c)) {}
  KOKKOS_INLINE_FUNCTION
  constexpr iterator begin() const noexcept {
    return p;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr iterator end() const noexcept {
    return q;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  KOKKOS_INLINE_FUNCTION
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  KOKKOS_INLINE_FUNCTION
  constexpr reference front() const {
    return *begin();
  }

  KOKKOS_INLINE_FUNCTION
  constexpr reference back() const {
    return end()[-1];
  }

  KOKKOS_INLINE_FUNCTION
  constexpr reference operator[](size_type i) const {
    return begin()[i];
  }

  KOKKOS_INLINE_FUNCTION
  constexpr pointer data() const noexcept {
    return begin();
  }

  // FIXME: Spurious overflow for extremely large ranges
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const noexcept {
    return end() - begin();
  }

  KOKKOS_INLINE_FUNCTION
  constexpr size_type size_bytes() const noexcept {
    return sizeof(element_type) * size();
  }

  KOKKOS_INLINE_FUNCTION
  constexpr bool empty() const noexcept {
    return begin() == end();
  }

  KOKKOS_INLINE_FUNCTION
  constexpr span first(size_type n) const {
    return {begin(), n};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr span last(size_type n) const {
    return {end() - n, n};
  }
  KOKKOS_INLINE_FUNCTION
  constexpr span subspan(size_type i, size_type n = -1) const {
    return {begin() + i, n == size_type(-1) ? size() - i : n};
  }

private:
  pointer p, q;
};

template<class C>
span(C &) -> span<typename C::value_type>;
template<class C>
span(const C &) -> span<const typename C::value_type>;

/// Copy a span into a std::vector.
template<class T>
auto
to_vector(span<T> s) {
  // Work around GCC<8 having no deduction guide for vector:
  return std::vector<typename span<T>::value_type>(s.begin(), s.end());
}
