//
// Created by Masshiro on 2022/9/21.
//

#ifndef CARDINALITYMOMENTESTIMATOR_HIERARCHYUTILS_HPP
#define CARDINALITYMOMENTESTIMATOR_HIERARCHYUTILS_HPP

#include <cstdint>
#include <immintrin.h>
#include <cmath>
#include <bitset>

namespace hierarchy{
//--------------------------
// Calculate moments: G-sum functions
    double G_sum(double base, uint8_t power){
        if (power == 0) return 1;
        if (power == 1) return base;
        return std::pow(base, power);
    }

    double G_log_sum(double x, uint8_t power) {
        return std::pow(std::log(x), power);
    }

    double G_entropy(double x){
//    if(x < 0)
//        x = 1;
        return x * std::log(x);
    }

    double G_entropy_abs(double x){
        x = std::fabs(x);
        return x * std::log(x);
    }

    bool get_hash_bit(uint32_t hash_val, uint8_t index){
        std::bitset<32> hash_value = hash_val;
//    return hash_value[index-1];
        return hash_value[31-index];
    }

    template<class KEY, class VAL>
    struct CompareByValue{
        bool operator() (const std::pair<KEY, VAL>& p1, const std::pair<KEY, VAL>& p2){
            return p1.second > p2.second;
        }
    };

}   // The end of namespace


#endif //CARDINALITYMOMENTESTIMATOR_HIERARCHYUTILS_HPP
