//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_ARTHIMETIC_HPP
#define CUBBYDNN_ARTHIMETIC_HPP
#include <cmath>
#include "../Types/TypeCast.hpp"
#include "../Types/bFloat16.hpp"
namespace CubbyDNN
{
class CpuArthimatic
{
    /**
     * These templates are for types fits into traditional
     * cmath libraries

     */
    template <typename T>
    static T Exponential(T input);

    template <typename T>
    static T Sin(T input);

    template <typename T>
    static T Cos(T input);

    template <typename T>
    static T Tan(T input);

    template <typename T>
    static T Tanh(T input);

    template <typename T>
    static T Arctan(T input);

    template <typename T>
    static T Square(T input);

    template <typename T>
    static T Sec(T input);

    /**
     * Special implementations for bfloat16 types
     * Internally converts bfloat16 to float16
     */
    static bFloat16 Exponential(bFloat16 input);

    static bFloat16 Sin(bFloat16 input);

    static bFloat16 Cos(bFloat16 input);

    static bFloat16 Tan(bFloat16 input);

    static bFloat16 Tanh(bFloat16 input);

    static bFloat16 Arctan(bFloat16 input);

    static bFloat16 Square(bFloat16 input);
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_ARTHIMETIC_HPP
