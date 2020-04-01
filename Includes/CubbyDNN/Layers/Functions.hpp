

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

 public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace) override
    {
        auto* output = &input;
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
    T& backward(T& input, bool inplace) override
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
    T& forward(T& input, bool inplace) override
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        sigmoid_result = *output = 1 / (1 + blaze::exp(-input));
        return *output;
    }
    T& backward(T& input, bool inplace) override
    {
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        *output = input % sigmoid_result % (1 - sigmoid_result);
        return *output;
    }
};

template<NumberSystem N>
class CrossEntropyWithSoftmax{
    using MT = typename NumToMat<N>::type;
    using NT = typename NumToType<N>::type;
private:
    MT grad;
public:
    void softmax_forward(MT& target){
        for(size_t i=0;i<target.rows();++i){
            NT max_val = target(i,0);
            NT sum_val = 0;
            for(size_t j=1;j<target.columns();++j){
                if(max_val < target(i,j)) max_val = target(i,j);
            }
            for(size_t j=0;j<target.columns();++j){
                target(i,j) = blaze::exp(target(i,j)-max_val);
                sum_val += target(i,j);
            }
            for(size_t j=0;j<target.columns();++j){
                (target(i,j)) /= sum_val;
            }
        }
    }

    NT forward(MT& input, MT& target){
        NT batch_size = input.rows();
        NT res = 0;
        softmax_forward(input);
        grad = input;
        if(target.columns() == input.columns()){
            res = blaze::sum(-blaze::log(input % target));
            grad -= target;
            grad /= batch_size;
        }else {
            for(size_t i=0;i<target.rows();++i){
                res += -blaze::log(input(i,static_cast<size_t>(target(i, 0))));
                grad(i,static_cast<size_t>(target(i, 0)))-=1;
            }
            grad /= batch_size;
        }
        return res;
    }

    MT backward(){
        return grad;
    }
};

template <NumberSystem N>
class MSE
{
    using MT = typename NumToMat<N>::type;
    using NT = typename NumToType<N>::type;

 private:
    MT back_prop;

 public:
    NT forward(MT& input, MT& target)
    {
        back_prop = input - target;
        NT batch_size = input.rows();
        NT output = blaze::sum(blaze::pow(back_prop, 2)) / batch_size;
        back_prop *= 2 / batch_size;
        return output;
    }
    MT backward()
    {
        return back_prop;
    }
};

template <NumberSystem N>
class ReLU6 : public Layer<N>{
    using T = typename NumToMat<N>::type;

private:
    blaze::CompressedMatrix<bool> mask;

public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());

        mask.resize(output->rows(), output->columns(), false);

        for (size_t i = 0UL; i < output->rows(); ++i)
        {
            for (size_t j = 0UL; j < output->columns(); ++j)
            {
                if (input(i, j) > 0 && input(i,j) <=6)
                {
                    if (!inplace)
                        (*output)(i, j) = input(i, j);
                    mask(i, j) = true;
                }
                else if(input(i,j)>6)
                {
                    (*output)(i, j) = 6;
                }else{
                    (*output)(i, j) = 0;
                }
            }
        }

        return *output;
    }
    T& backward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        *output = mask % input;
        return *output;
    }
};

template <NumberSystem N>
class LeakyReLU: public Layer<N>{
    using T = typename NumToMat<N>::type;
    using NT = typename NumToType<N>::type;
private:
    T mask;
public:
    NT leaky_val;

    LeakyReLU(NT l_val=0.01): leaky_val(l_val) {}

    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());

        mask.resize(output->rows(), output->columns());

        for (size_t i = 0UL; i < output->rows(); ++i)
        {
            for (size_t j = 0UL; j < output->columns(); ++j)
            {
                if (input(i, j) > 0)
                {
                    if (!inplace)
                        (*output)(i, j) = input(i, j);
                    mask(i, j) = 1;
                }
                else{
                    (*output)(i, j) *= leaky_val;
                    mask(i, j) = leaky_val;
                }
            }
        }

        return *output;
    }
    T& backward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        *output = mask % input;
        return *output;
    }
};

template <NumberSystem N>
class Tanh: public Layer<N>{
    using T = typename NumToMat<N>::type;
private:
    T tan_val;
public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());
        auto pos_exp = blaze::exp(input), neg_exp = blaze::exp(-input);

        tan_val = (pos_exp - neg_exp) / (pos_exp + neg_exp);
        return (*output) = tan_val;
    }
    T& backward(T& input, bool inplace) override{
        auto* output = &input;
        if (!inplace)
            output = new T(input.rows(), input.columns());

        return *output = (1 - blaze::pow(tan_val, 2)) % input;
    }
};

template <NumberSystem N>
class ELU : public Layer<N>{
    using T = typename NumToMat<N>::type;
private:
    T tan_val;
public:
    using Layer<N>::forward;
    using Layer<N>::backward;
    T& forward(T& input, bool inplace) override{

    }

};

}  // namespace CubbyDNN
#endif  // CUBBYDNN_FUNCTIONS_HPP
