//
// Created by Yoo Geonhui on 2020/03/15.
//

#ifndef CUBBYDNN_INITIALIZERS_HPP
#define CUBBYDNN_INITIALIZERS_HPP

#include <blaze/Blaze.h>
#include <CubbyDNN/NumberSystem.hpp>
#include <random>

namespace CubbyDNN{
    template<NumberSystem N>
    class Initializer{
        using MT = typename NumToMat<N>::type;
        using T = typename NumToType<N>::type;
        using RV = blaze::DynamicVector<T, blaze::rowVector>;
        using CV = blaze::DynamicVector<T, blaze::columnVector>;
    public:
        virtual void initialize(MT& param) = 0;
        virtual void initialize(RV& param) = 0;
        virtual void initialize(CV& param) = 0;
    };

    template<NumberSystem N>
    class NormalDistribution: Initializer<N>{
        using MT = typename NumToMat<N>::type;
        using T = typename NumToType<N>::type;
        using RV = blaze::DynamicVector<T, blaze::rowVector>;
        using CV = blaze::DynamicVector<T, blaze::columnVector>;
    private:
        T mean, dist;
    public:
        NormalDistribution(T mean, T dist): mean(mean), dist(dist){}
        void initialize(MT& param) override{
            std::default_random_engine generator(std::random_device{}());
            std::normal_distribution<T> distribution(mean, dist);

            for(size_t i=0;i<param.rows();i++){
                for(size_t j=0;j<param.columns();j++){
                    param = distribution(generator);
                }
            }
        }
        void initialize(RV& param) override{
            std::default_random_engine generator(std::random_device{}());
            std::normal_distribution<T> distribution(mean, dist);

            for(auto& i: param){
                param = distribution(generator);
            }
        }
        void initialize(CV& param) override{
            std::default_random_engine generator(std::random_device{}());
            std::normal_distribution<T> distribution(mean, dist);

            for(auto& i: param){
                param = distribution(generator);
            }
        }
    };

    template<NumberSystem N>
    class Xavier: Initializer<N>{
        using MT = typename NumToMat<N>::type;
        using T = typename NumToType<N>::type;
        using RV = blaze::DynamicVector<T, blaze::rowVector>;
        using CV = blaze::DynamicVector<T, blaze::columnVector>;
    public:
        void initialize(MT& param) override{
            T dist = (T)std::sqrt((double)2/(param.rows() + param.columns()));

            auto normal_dist = NormalDistribution<N>(0, dist);
            normal_dist.initialize(param);
        }
        void initialize(RV& param) override{

        }
        void initialize(CV& param) override{

        }
    };
};

#endif //CUBBYDNN_INITIALIZERS_HPP
