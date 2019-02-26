//
// Created by jwkim98 on 19/02/18.
//

#ifndef CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
#define CUBBYDNN_ACTIVATIONFUNCTIONS_HPP

#include <cmath>
#include <functional>
#include "cubbydnn/Arthimetics/Arithmetic.hpp"

namespace CubbyDNN
{
template <typename T>
class Relu
{
 public:
    T Forward(T input)
    {
        return (input > 0) ? input : 0;
    }

    T Backward(T input)
    {
        return (input > 0) ? 1 : 0;
    }
};

template <typename T>
class SmoothRelu
{
 public:
    T Forward(T input)
    {
        /// T should have conversion from double
        return CpuArithmetic::Log(1 + input);
    }

    T Backward(T input)
    {
        return 1/Log(1+input);
    }
};

template <typename T>
class LeakyRelu
{
 public:
    explicit LeakyRelu(T alpha) : alpha(alpha)
    {
    }

    T Forward(T input)
    {
        return (input > 0) ? input : alpha * input;
    }

    T Backward(T input)
    {
        return (input > 0) ? 1 : alpha;
    }

 private:
    T alpha = 0;
};

template <typename T>
class ELU
{
 public:
    explicit ELU(T alpha) : alpha(alpha)
    {
    }

    T Forward(T input)
    {
        return (input > 0) ? input : alpha * (CpuArithmetic::Exponential(input) - 1);
    }

    T Backward(T input)
    {
        return (input > 0) ? 1 : alpha*CpuArithmetic::Exponential(input);
    }
 private:
    T alpha;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
