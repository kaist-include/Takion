#ifndef CUBBYDNN_OPTIMIZER_HPP
#define CUBBYDNN_OPTIMIZER_HPP


#include <blaze/Blaze.h>
#include <CubbyDNN/NumberSystem.hpp>

namespace CubbyDNN
{
template<NumberSystem N> 
class SGD
{
using T = typename NumToType<N>::type;
using MT = typename NumToMat<N>::type;
using VT = typename blaze::DynamicVector<T, blaze::rowVector>;

private:
T lr;

public:
SGD(T lr) : lr(lr) {}
inline void update(MT& parameter, MT& grad)
{
    parameter -= lr * grad;
}
inline void update(VT& parameter, VT& grad)
{
    parameter -= lr * grad;
}
};
};  // namespace CubbyDNN

#endif