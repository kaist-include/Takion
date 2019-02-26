//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_SOFTMAX_HPP
#define CUBBYDNN_SOFTMAX_HPP
#include <cmath>
namespace CubbyDNN{
    template<typename T1, typename ...T>
    class SoftMax{
    public:
        explicit SoftMax(T ...args){
            exponentialSum(args...);
        }

        T1 operator()(T1 input){
            return static_cast<T1>(exp(input))/m_exponentialSum;
        }

    private:
        T1 exponentialSum(T1 arg){
            return static_cast<T1>(exp(arg));
        }

        T1 exponentialSum(T1 arg, T... args){
            return static_cast<T1>(exp(arg)) + exponentialSum(args...);
        }

        T1 m_exponentialSum;
    };
}

#endif //CUBBYDNN_SOFTMAX_HPP
