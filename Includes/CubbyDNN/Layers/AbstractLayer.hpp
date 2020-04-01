#ifndef CUBBYDNN_LAYER_HPP
#define CUBBYDNN_LAYER_HPP

#include <blaze/Blaze.h>
#include <CubbyDNN/Optimizers.hpp>
#include <CubbyDNN/NumberSystem.hpp>

namespace CubbyDNN
{
template <NumberSystem N>
class Layer
{
 public:
    using T = typename NumToMat<N>::type;
    inline T& forward(T& input)
    {
        return forward(input, false);
    }
    inline T& backward(T& input)
    {
        return backward(input, false);
    }
    virtual T& forward(T& input, bool inplace) = 0;
    virtual T& backward(T& input, bool inplace) = 0;
    virtual void apply_grad(Optimizer<N>&){}
};
}  // namespace CubbyDNN
#endif  // CUBBYDNN_LAYER_HPP
