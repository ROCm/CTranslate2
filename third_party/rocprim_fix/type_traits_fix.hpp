#ifndef ROCPRIM_TYPE_TRAITS_FIX_HPP
#define ROCPRIM_TYPE_TRAITS_FIX_HPP

// Workaround for ROCm 6.3.0 rocprim bug in type_traits.hpp
// The original header has invalid template specialization inside struct
// This file provides a corrected version using if constexpr

#include <type_traits>

namespace rocprim {

// Forward declare rocprim::tuple types we need
template<class... Args> struct tuple;
template<size_t I, class T> struct tuple_element;
template<size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

// Fixed version using if constexpr instead of template specialization
template<class T>
struct is_tuple_of_references
{
    static_assert(sizeof(T) == 0, "is_tuple_of_references is only implemented for rocprim::tuple");
};

template<class... Args>
struct is_tuple_of_references<::rocprim::tuple<Args...>>
{
private:
    template<size_t Index>
    static constexpr bool is_tuple_of_references_impl()
    {
        if constexpr (Index >= sizeof...(Args)) {
            return true;
        } else {
            using tuple_t   = ::rocprim::tuple<Args...>;
            using element_t = ::rocprim::tuple_element_t<Index, tuple_t>;
            return std::is_reference<element_t>::value && is_tuple_of_references_impl<Index + 1>();
        }
    }

public:
    static constexpr bool value = is_tuple_of_references_impl<0>();
};

} // namespace rocprim

#endif // ROCPRIM_TYPE_TRAITS_FIX_HPP
