//
// Created by jwkim98 on 19/02/18.
//

#ifndef CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
#define CUBBYDNN_ACTIVATIONFUNCTIONS_HPP

#include <cmath>
#include <functional>
#include "../Arthimetics/Arthimetic.hpp"

namespace CubbyDNN
{
template <typename T>
class Relu
{
 public:
    T operator()(T input)
    {
        return (input > 0) ? input : 0;
    }
};

template <typename T>
class SmoothRelu
{
 public:
    T operator()(T input)
    {
        /// T should have conversion from double
        return static_cast<T>(log(1 + (input)));
    }
};

template <typename T>
class LeakyRelu
{
 public:
    explicit LeakyRelu(T alpha) : alpha(alpha)
    {
    }

    T operator()(T input)
    {
        return (input > 0) ? input : alpha * input;
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

    T operator()(T input)
    {
        return (input > 0) ? input : alpha * (static_cast<T>(exp(input) - 1));
    }

 private:
    T alpha;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
