//
// Created by jwkim98 on 19/02/18.
//

#ifndef CUBBYDNN_SIGMOIDS_HPP
#define CUBBYDNN_SIGMOIDS_HPP
#include <cmath>
namespace CubbyDNN
{
template <typename T>
class Logistic
{
 public:
    explicit Logistic(T param = 1) : m_param(param)
    {
    }

    T operator()(T input)
    {
        if (m_param == 1)
            return static_cast<T>(1 / (1 + exp(input)));
        else
        {
            return static_cast<T>(1 / pow(1 + exp(input), m_param));
        }
    }

 private:
    T m_param;
};

template <typename T>
class Tanh
{
 public:
    T operator()(T input)
    {
        return static_cast<T>(tanh(input));
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SIGMOIDS_HPP
