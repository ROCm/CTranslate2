#pragma once

// ROCm 6.3.0 rocprim/type_traits.hpp bug workaround
// The original header has invalid template specialization inside struct (lines 298-302)
// This header prevents the buggy type_traits.hpp from being included by defining its guard
// and provides a fixed version of the problematic template

#ifndef ROCPRIM_TYPE_TRAITS_HPP_
#define ROCPRIM_TYPE_TRAITS_HPP_

#include <type_traits>

// Forward declare rocprim types we need
namespace rocprim {
    template<class... Args> struct tuple;
    template<size_t I, class T> struct tuple_element;
    template<size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

    #ifndef ROCPRIM_HOST_DEVICE
    #define ROCPRIM_HOST_DEVICE __host__ __device__
    #endif
}

namespace rocprim {
namespace detail {

// Fixed version of is_tuple_of_references using if constexpr instead of template specialization
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
    ROCPRIM_HOST_DEVICE static constexpr bool is_tuple_of_references_impl()
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

// Include rest of type_traits content (float_bit_mask onwards)
template<class Key>
struct float_bit_mask;

template<>
struct float_bit_mask<float>
{
    static constexpr uint32_t value = 0x80000000;
    using bit_type = uint32_t;
};

template<>
struct float_bit_mask<double>
{
    static constexpr uint64_t value = 0x8000000000000000;
    using bit_type = uint64_t;
};

template<>
struct float_bit_mask<__half>
{
    static constexpr uint16_t value = 0x8000;
    using bit_type = uint16_t;
};

template<>
struct float_bit_mask<hip_bfloat16>
{
    static constexpr uint16_t value = 0x8000;
    using bit_type = uint16_t;
};

} // namespace detail
} // namespace rocprim

#endif // ROCPRIM_TYPE_TRAITS_HPP_
