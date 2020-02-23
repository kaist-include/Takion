

#ifndef CUBBYDNN_FUNCTIONS_HPP
#define CUBBYDNN_FUNCTIONS_HPP

#include <blaze/Blaze.h>
#include <CubbyDNN/Layers/AbstractLayer.hpp>

namespace CubbyDNN
{
template <NumberSystem N>
class ReLU : public Layer<N>
{
    using T = typename NumToMat<N>::type;

 private:
    blaze::CompressedMatrix<bool> mask;
    bool is_passed = false;

 public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace)
    {
        auto* output = &input;
        is_passed = true;
        if (!inplace)
            output = new T(input.rows(), input.columns());

        mask.resize(output->rows(), output->columns(), false);

        for (size_t i = 0UL; i < output->rows(); ++i)
        {
            for (size_t j = 0UL; j < output->columns(); ++j)
            {
                if (input(i, j) > 0)
                {
                    if (!inplace)
                        (*output)(i, j) = input(i, j);
                    mask(i, j) = true;
                }
                else
                {
                    (*output)(i, j) = 0;
                }
            }
        }

        return *output;
    }
    T& backward(T& input, bool inplace)
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());

        *output = mask % input;
        return *output;
    }
};

template <NumberSystem N>
class Sigmoid : public Layer<N>
{
    using T = typename NumToMat<N>::type;

 private:
    T sigmoid_result;

 public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace)
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        sigmoid_result = *output = 1 / (1 + blaze::exp(-input));
        return *output;
    }
    T& backward(T& input, bool inplace)
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        *output = input % sigmoid_result % (1 - sigmoid_result);
        return *output;
    }
};

template <NumberSystem N>
class MSE
{
    using MT = typename NumToMat<N>::type;
    using T = typename NumToType<N>::type;

 private:
    MT subtraction;
    T batch_size;

 public:
    T forward(MT& input, MT& target)
    {
        subtraction = input - target;
        batch_size = input.rows();
        T output = blaze::sum(blaze::pow(subtraction, 2)) / batch_size;
        return output;
    }
    MT backward()
    {
        return subtraction * 2 / batch_size;
    }
};
}  // namespace CubbyDNN
#endif  // CUBBYDNN_FUNCTIONS_HPP
