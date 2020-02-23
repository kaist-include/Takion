#ifndef CUBBYDNN_NUMBER_SYSTEM_HPP
#define CUBBYDNN_NUMBER_SYSTEM_HPP

#include <blaze/Blaze.h>
namespace CubbyDNN
{
enum class NumberSystem
{
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64
};

template <NumberSystem T>
struct NumToType;

template <>
struct NumToType<NumberSystem::Int8>
{
    using type = blaze::int8_t;
};

template <>
struct NumToType<NumberSystem::Int16>
{
    using type = blaze::int16_t;
};

template <>
struct NumToType<NumberSystem::Int32>
{
    using type = blaze::int32_t;
};

template <>
struct NumToType<NumberSystem::Int64>
{
    using type = blaze::int64_t;
};

template <>
struct NumToType<NumberSystem::Float32>
{
    using type = float;
};

template <NumberSystem T>
struct NumToMat;

template <>
struct NumToMat<NumberSystem::Int8>
{
    using type = blaze::DynamicMatrix<blaze::int8_t>;
};

template <>
struct NumToMat<NumberSystem::Int16>
{
    using type = blaze::DynamicMatrix<blaze::int16_t>;
};

template <>
struct NumToMat<NumberSystem::Int32>
{
    using type = blaze::DynamicMatrix<blaze::int32_t>;
};

template <>
struct NumToMat<NumberSystem::Int64>
{
    using type = blaze::DynamicMatrix<blaze::int64_t>;
};

template <>
struct NumToMat<NumberSystem::Float32>
{
    using type = blaze::DynamicMatrix<float, blaze::rowMajor>;
};
}  // namespace CubbyDNN

#endif