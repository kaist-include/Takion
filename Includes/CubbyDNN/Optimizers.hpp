#ifndef CUBBYDNN_OPTIMIZER_HPP
#define CUBBYDNN_OPTIMIZER_HPP

#include <blaze/Blaze.h>
#include <CubbyDNN/NumberSystem.hpp>

namespace CubbyDNN
{

    template<NumberSystem N>
    class Optimizer{
        using T = typename NumToType<N>::type;
        using MT = typename NumToMat<N>::type;
        using VT = typename blaze::DynamicVector<T, blaze::rowVector>;
    public:
        virtual void update(MT& parameter, MT& grad) = 0;
        virtual void update(VT& parameter, VT& grad) = 0;
    };

    template<NumberSystem N>
    class SGD: public Optimizer<N>
    {
        using T = typename NumToType<N>::type;
        using MT = typename NumToMat<N>::type;
        using VT = typename blaze::DynamicVector<T, blaze::rowVector>;

    private:
        T lr;

    public:
        SGD(T lr) : lr(lr) {}

        void update(MT& parameter, MT& grad) override
        {
            parameter -= lr * grad;
        }

        void update(VT& parameter, VT& grad) override
        {
            parameter -= lr * grad;
        }
    };

    template<NumberSystem N>
    class Nesterov: public Optimizer<N>
    {
        using T = typename NumToType<N>::type;
        using MT = typename NumToMat<N>::type;
        using VT = typename blaze::DynamicVector<T, blaze::rowVector>;

    private:
        T lr;

    public:
        Nesterov(T lr) : lr(lr) {}

        void update(MT& parameter, MT& grad) override
        {
            parameter -= lr * grad;
        }

        void update(VT& parameter, VT& grad) override
        {
            parameter -= lr * grad;
        }
    };

    template<NumberSystem N>
    class Adam: public Optimizer<N>{
        using T = typename NumToType<N>::type;
        using MT = typename NumToMat<N>::type;
        using VT = typename blaze::DynamicVector<T, blaze::rowVector>;

    private:
        T lr, beta1, beta2, t, eps;
        MT m_matrix, v_matrix;
        VT m_vector, v_vector;
    public:
        Adam(T lr, T beta1=0.9, T beta2=0.999, T eps=1e-8): lr(lr), beta1(beta1), beta2(beta2), eps(eps){}

        void update(MT& parameter, MT& grad) override{
            if(m_matrix.rows() == 0){
                m_matrix = blaze::zero<T>(parameter.rows(), parameter.columns());
                v_matrix = blaze::zero<T>(parameter.rows(), parameter.columns());
            }
            m_matrix = beta1 * m_matrix + (1 - beta1) * grad;
            v_matrix = beta2 * v_matrix + (1 - beta2) * blaze::exp2(grad);

            MT m_bias_fixed = m_matrix / (1 - blaze::exp(beta1, t));
            MT v_bias_fixed = v_matrix / (1 - blaze::exp(beta2, t));

            parameter = parameter - lr * m_bias_fixed / (blaze::sqrt(v_bias_fixed) + eps);
        }

        void update(VT& parameter, VT& grad) override{

        }
    };

};  // namespace CubbyDNN

#endif