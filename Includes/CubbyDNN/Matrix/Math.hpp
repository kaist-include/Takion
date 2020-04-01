#ifndef CUBBYDNN_MATH_HPP
#define CUBBYDNN_MATH_HPP

#include <CubbyDNN/Matrix/Matrix.hpp>
#include <cmath>

namespace CubbyDNN {
    template<typename T>
    Matrix<T> sin(Matrix<T>& matrix) {
        matrix.for_each([](T i){
            return std::sin(i);
        });
    }

    template<typename T>
    Matrix<T> cos(Matrix<T>& matrix) {
        matrix.for_each([](T i){
            return std::cos(i);
        });
    }

    template<typename T>
    Matrix<T> tan(Matrix<T>& matrix) {
        matrix.for_each([](T i){
            return std::tan(i);
        });
    }

    template<typename T>
    Matrix<T> exp(Matrix<T>& matrix){
        matrix.for_each([](T i){
            return std::exp(i);
        });
    }

    template<typename T>
    Matrix<T> log(Matrix<T>& matrix){
        matrix.for_each([](T i){
            return std::log(i);
        });
    }
}
#endif //CUBBYDNN_MATRIXMATH_HPP
