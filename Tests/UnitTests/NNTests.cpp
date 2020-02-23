#include <CubbyDNN/Layers/Functions.hpp>
#include <CubbyDNN/Layers/NN.hpp>
#include <CubbyDNN/NumberSystem.hpp>
#include <CubbyDNN/Optimizers.hpp>
#include <iostream>
#include "doctest.h"

namespace CubbyDNN
{
const NumberSystem N = NumberSystem::Float32;
TEST_CASE("NN simple linear")
{
    size_t print_iter = 100;

    std::vector<Layer<N>*> Sequential;
    Layer<N>* layer1 = new NN<N>(1, 1, true);
    Sequential.push_back(layer1);

    MSE<N> loss;
    SGD<N> optimizer(0.005);

    auto dataset_linear_X = NumToMat<N>::type({ { 1 }, { 4 }, { 7 } });
    auto dataset_linear_Y = NumToMat<N>::type({ { 2 }, { 8 }, { 14 } });

    for (size_t epoch = 0; epoch < 1000; ++epoch)
    {
        auto X = dataset_linear_X;
        for (auto layer : Sequential)
        {
            layer->forward(X, true);
        }
        NumToType<N>::type loss_val = loss.forward(X, dataset_linear_Y);
        if (epoch % print_iter == 0)
        {
            MESSAGE("loss: " << loss_val);
        }
            typename NumToMat<N>::type back = loss.backward();
        for (auto i = Sequential.rbegin(); i != Sequential.rend(); i++)
        {
            (*i)->backward(back, true);
            (*i)->apply_grad(optimizer);
        }
    }

    auto X = dataset_linear_X;
    for (auto layer : Sequential)
    {
        layer->forward(X, true);
    }
    NumToType<N>::type res = loss.forward(X, dataset_linear_Y);

    CHECK_LE(res, 1);
}
}  // namespace CubbyDNN