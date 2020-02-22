

#ifndef CUBBYDNN_ACTIVATIONS_HPP
#define CUBBYDNN_ACTIVATIONS_HPP

#include <CubbyDNN/Layers/AbstractLayer.hpp>
#include <blaze/Blaze.h>

namespace CubbyDNN{
    template<NumberSystem N>
    class ReLU : public Layer<N>{
        using T = typename NumToMat<N>::type;
    private:
        blaze::CompressedMatrix<bool> mask;
        bool is_passed = false;
    public:
        using Layer<N>::forward;
        using Layer<N>::backward;
        T& forward(T& input, bool inplace){
            auto* output = &input;
            is_passed = true;
            if(!inplace) output = new T(input.rows(), input.columns());

            mask.resize(output->rows(), output->columns(), false);

            for(size_t i = 0UL; i<output->rows(); ++i) {
                for(size_t j = 0UL; j<output->columns();++j){
                    if(input(i,j)>0){
                        if(!inplace) (*output)(i,j) = input(i,j);
                        mask(i,j) = true;
                    }else{
                        (*output)(i,j) = 0;
                    }
                }
            }

            return *output;
        }
        T& backward(T& input, bool inplace) {
            auto* output = &input;
            if(!inplace) output = new T(input.rows(), input.columns());

            *output = mask % input;
            return *output;
        }
    };

    template<NumberSystem N>
    class Sigmoid : public Layer<N>{
        using T = typename NumToMat<N>::type;
    private:
        T sigmoid_result;
    public:
        using Layer<N>::forward;
        using Layer<N>::backward;
        T& forward(T& input, bool inplace){
            auto* output = &input;
            if(!inplace) output = new T(input.rows(), input.columns());
            sigmoid_result = *output = 1/(1+blaze::exp(-input));
            return *output;
        }
        T& backward(T& input, bool inplace) {
            auto* output = &input;
            if(!inplace) output = new T(input.rows(), input.columns());
            *output = input % sigmoid_result % (1-sigmoid_result);
            return *output;
        }
    };

}

#endif //CUBBYDNN_ACTIVATIONS_HPP
