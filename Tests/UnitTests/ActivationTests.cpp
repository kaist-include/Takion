#include <CubbyDNN/Layers/Functions.hpp>
#include "doctest.h"
#include <blaze/Blaze.h>
#include <cmath>

namespace CubbyDNN {


    TEST_CASE ("ReLU") {
        auto test_matrix1 = blaze::DynamicMatrix<float>({{ 3.1, 6.4 }, { -0.9, -1.2 }, { 4.8, 0.6 }});
        auto test_matrix2 = blaze::DynamicMatrix<float>({{ 3.1, 6.4 }, { 0, 0 }, { 4.8, 0.6 }});
        auto test_matrix3 = blaze::DynamicMatrix<float>({{2.5, -1.1}, {5, 4}, {4, 3}});
        auto test_matrix4 = blaze::DynamicMatrix<float>({{2.5, -1.1}, {0, 0}, {4, 3}});

        ReLU<NumberSystem::Float32> ReLULayer;
        CHECK_EQ(ReLULayer.forward(test_matrix1), test_matrix2);
        CHECK_EQ(ReLULayer.backward(test_matrix3), test_matrix4);
        ReLULayer.forward(test_matrix1, true);
        CHECK_EQ(test_matrix1, test_matrix2);
        ReLULayer.backward(test_matrix3, true);
        CHECK_EQ(test_matrix3, test_matrix4);
    }

    TEST_CASE("Sigmoid"){
        auto test_matrix1 = NumToMat<NumberSystem::Float32>::type({{ 3.1, 0 }, { -0.9, -1.2 }, { 4.8, 0.6 }});
        auto element_wise_sigmoid = [](const NumToMat<NumberSystem::Float32>::type& input) -> NumToMat<NumberSystem::Float32>::type {
            auto res = input;
            for(size_t i = 0UL; i<res.rows(); ++i){
                for(size_t j = 0UL; j<res.columns();++j){
                    res(i,j) = 1/(1+std::exp((float)-res(i,j)));
                }
            }
            return res;
        };
        auto test_matrix2 = element_wise_sigmoid(test_matrix1);
        auto test_matrix3 = NumToMat<NumberSystem::Float32>::type({{ 1, 1},{1,1},{1,1}});
        Sigmoid<NumberSystem::Float32> SigmoidLayer;
        CHECK_EQ(SigmoidLayer.forward(test_matrix1), test_matrix2);
        CHECK_EQ(SigmoidLayer.backward(test_matrix3), test_matrix2 % (1-test_matrix2));
    }

    TEST_CASE("ReLU6"){
        auto test_matrix1 = blaze::DynamicMatrix<float>({{ 3.1, 6.4 }, { -0.9, -1.2 }, { 6.8, 0.6 }});
        auto test_matrix2 = blaze::DynamicMatrix<float>({{ 3.1, 6 }, { 0, 0 }, { 6, 0.6 }});
        auto test_matrix3 = blaze::DynamicMatrix<float>({{2.5, -1.1}, {5, 4}, {4, 3}});
        auto test_matrix4 = blaze::DynamicMatrix<float>({{2.5, 0}, {0, 0}, {0, 3}});

        ReLU6<NumberSystem::Float32> ReLU6Layer;

        CHECK_EQ(ReLU6Layer.forward(test_matrix1), test_matrix2);
        CHECK_EQ(ReLU6Layer.backward(test_matrix3), test_matrix4);

        ReLU6Layer.forward(test_matrix1, true);

        CHECK_EQ(test_matrix1, test_matrix2);

        ReLU6Layer.backward(test_matrix3, true);

        CHECK_EQ(test_matrix3, test_matrix4);
    }

    TEST_CASE("LeakyReLU"){
        auto test_matrix1 = blaze::DynamicMatrix<float>({{ 3.1, 6.4 }, { -0.9, -1.2 }, { 6.8, 0.6 }});
        auto test_matrix2 = blaze::DynamicMatrix<float>({{ 3.1, 6.4 }, { -0.009, -0.012 }, { 6.8, 0.6 }});
        auto test_matrix3 = blaze::DynamicMatrix<float>({{2.5, -1.1}, {5, 4}, {4, 3}});
        auto test_matrix4 = blaze::DynamicMatrix<float>({{2.5, -1.1}, {0.05, 0.04}, {4, 3}});

        LeakyReLU<NumberSystem::Float32> LeakyReLULayer;

        CHECK_EQ(LeakyReLULayer.forward(test_matrix1), test_matrix2);
        CHECK_EQ(LeakyReLULayer.backward(test_matrix3), test_matrix4);

        LeakyReLULayer.forward(test_matrix1, true);

        CHECK_EQ(test_matrix1, test_matrix2);

        LeakyReLULayer.backward(test_matrix3, true);

        CHECK_EQ(test_matrix3, test_matrix4);
    }
}