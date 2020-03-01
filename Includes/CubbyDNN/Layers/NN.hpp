#ifndef CUBBYDNN_NN_HPP
#define CUBBYDNN_NN_HPP

#include <CubbyDNN/Layers/AbstractLayer.hpp>
#include <CubbyDNN/NumberSystem.hpp>
#include <CubbyDNN/Optimizers.hpp>
#include <random>

namespace CubbyDNN
{
template <NumberSystem N>
class NN : public Layer<N>
{
    using T = typename NumToMat<N>::type;

 private:
    T weight, dW;
    typename blaze::DynamicVector<typename NumToType<N>::type, blaze::rowVector> dB, bias;
    size_t in, out;
    bool is_bias;
    T x;

 public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    NN(size_t in_dim, size_t out_dim, bool is_bias)
        : in(in_dim), out(out_dim), is_bias(is_bias)
    {
        double xavier_val = std::sqrt(2 / static_cast<double>(in + out));

        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> distribution(0.0, xavier_val);

        weight.resize(in, out);
        bias.resize(out);

        for (size_t i = 0; i < in; ++i)
        {
            for (size_t j = 0; j < out; j++)
            {
                weight(i, j) = distribution(generator);
            }
        }

        if (is_bias)
        {
            for (size_t j = 0; j < out; j++)
            {
                bias[j] = distribution(generator);
            }
        }
    }

    T& forward(T& input, bool inplace) override
    {
        auto* output = &input;
        x = blaze::trans(input);
        if (!inplace)
            output = new T(input.rows(), out);
        *output = input * weight;
        if (is_bias)
            for (size_t i = 0UL; i < (*output).rows(); ++i)
            {
                for (size_t j = 0UL; j < out; ++j)
                {
                    (*output)(i, j) += bias[j];
                }
            }
        return *output;
    }

    T& backward(T& input, bool inplace) override
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), in);
        dW = x * input;
        dB = blaze::sum<blaze::columnwise>(input);
        return *output = input * blaze::trans(weight);
    }

    void apply_grad(SGD<N>& optimizer) override
    {
        optimizer.update(weight, dW);
        if (is_bias)
        {
            optimizer.update(bias, dB);
        }
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_NN_HPP
