//#ifndef HISTOGRAM_TC_HPP
//#define HISTOGRAM_TC_HPP
//
//#include "fmt/core.h"
//#include "spdlog/spdlog.h"
//#include <array>
//#include <cmath>
//#include <cstdint>
//#include <cstdlib>
//#include <string>
//
//class HistogramTC {
//public:
//    static const std::size_t HISTOGRAM_LEN = 16;
//
//private:
//    static inline double Pr_Mj_eq_k(int k, int num_of_estimators, double estimation) {
//        if (k == 0) {
//            return std::pow((1 - 1.0 / num_of_estimators), estimation);
//        }
//
//        double a = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << k));
//        double b = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << (k - 1)));
//        double ret = std::pow(a, estimation) - std::pow(b, estimation);
//        return ret;
//    }
//
//    static inline double derivation_of_Pr_Mj_eq_k(int k, int num_of_estimators, double estimation) {
//        if (k == 0) {
//            double ret = std::pow((1 - 1.0 / num_of_estimators), estimation) * std::log2(1 - 1.0 / num_of_estimators);
//            return ret;
//        }
//        double a = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << k));
//        double b = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << (k - 1)));
//        double ret = std::pow(a, estimation) * std::log2(a) -
//                     std::pow(b, estimation) * std::log2(b);
//        return ret;
//    }
//
//    static double derivation_of_log_L(double estimation, std::array<uint32_t, HISTOGRAM_LEN> histogram, size_t histogram_len, int num_of_estimators) {
//        double ret = 0;
//        double d_pr_mj_eq_k = 0;
//        double pr_mj_eq_k = 0;
//        for (uint32_t i = 0; i < histogram_len; i++) {
//            uint32_t N_k = histogram[i];
//            if (N_k == 0) {
//                continue;
//            }
//            d_pr_mj_eq_k = derivation_of_Pr_Mj_eq_k(i, num_of_estimators, estimation);
//            pr_mj_eq_k = Pr_Mj_eq_k(i, num_of_estimators, estimation);
//            ret += N_k * (d_pr_mj_eq_k / pr_mj_eq_k);
//
//        }
//        return ret;
//    }
//
//    double MLE(double card_est) {
//        std::uint32_t bar = 0;
//        std::uint32_t smallest_value = 0;
//        for (std::size_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            bar = histogram.at(i);
//            if (bar > 0) {
//                smallest_value = i;
//                break;
//            }
//        }
//        double step_size = init_reg_num * alpha * (1 << smallest_value);
//        for (int iter = 0; iter < 20; iter++) {
//            card_est = card_est + step_size * derivation_of_log_L(card_est, histogram,HISTOGRAM_LEN, init_reg_num);
//        }
//        return card_est;
//    }
//
//public:
//    std::uint32_t init_reg_num;
//    double alpha;
//    bool bitmap_mle_enabler;
//    std::array<std::uint32_t, HISTOGRAM_LEN> histogram;
//    std::uint32_t base = 0;
//
//    HistogramTC(std::uint32_t reg_num, bool bitmap_mle_enabler = true)
//    : bitmap_mle_enabler(bitmap_mle_enabler) {
//        this->init_reg_num = reg_num;
//
//        if (reg_num == 16) {
//            alpha = 0.673;
//        } else if (reg_num == 32) {
//            alpha = 0.697;
//        } else if (reg_num == 64) {
//            alpha = 0.709;
//        } else {
//            alpha = 0.7213 / (1 + 1.079 / reg_num);
//        }
//
//        histogram = {reg_num, 0};
//        base = 0;
//    }
//
//    void reset() {
//        histogram = {this->init_reg_num, 0};
//        base = 0;
//    }
//
//    void update(std::size_t oldIdx, std::size_t newIdx) {
//        if (oldIdx > HISTOGRAM_LEN || oldIdx > newIdx) {
//            spdlog::error("access histogram: oldIdx:{}, newIdx:{}", oldIdx, newIdx);
//        }
//        if (newIdx >= HISTOGRAM_LEN) {
//            newIdx = HISTOGRAM_LEN;
//        }
//        std::uint32_t cur_val;
//        cur_val = histogram.at(oldIdx);
//        if (cur_val <= 0) {
//            return;
//        }
//
//        histogram.at(oldIdx) = cur_val - 1;
//        histogram.at(newIdx)++;
//    }
//
//    void move(std::size_t oldIdx, std::size_t newIdx) {
//        std::uint32_t cur_val;
//        cur_val = histogram.at(oldIdx);
//        histogram.at(oldIdx) = 0;
//        histogram.at(newIdx) = cur_val;
//    }
//
//    void updateBase(uint32_t delta_B) {
//        /* update the value of Base */
//        base += delta_B;
//        /* update the value of histogram */
//        for (std::uint32_t k = delta_B; k < HISTOGRAM_LEN; ++k) {
//            move(k, k - delta_B);
//        }
//    }
//
//    double getEstimate(std::uint32_t histogram_base) {
//        double power_sum = 0;
//        std::uint32_t bar;
//        for (std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            bar = histogram.at(i);
//            int tmp = i + histogram_base;
//            power_sum += bar * std::pow(2, -tmp);
//        }
//
//        double harmonic_avg = 1 / power_sum;
//        double card_est = alpha * init_reg_num * init_reg_num * harmonic_avg;
//
//        if (!bitmap_mle_enabler) {
//            return card_est;
//        }
//
//        if (card_est <= 2 * init_reg_num) {
//            uint32_t v = histogram.at(0);
//            if (v != 0) {
//                return (double)init_reg_num * log((double)init_reg_num/ v);
//            }
//            else {
//                return MLE(card_est);
//            }
//        }
//
//        else if (card_est > 5 * init_reg_num) {
//            return card_est;
//        }
//
//        return MLE(card_est);
//    }
//
//    /* find the minimum value stored in registers using histogram */
//    std::uint32_t getMinBar() {
//        std::uint32_t min_val = 0;
//        for (std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            uint32_t val =  histogram.at(i);
//            if (val != 0) {
//                min_val = i;
//                return min_val;
//            }
//        }
//        return 0;
//    }
//
//    /* FUNCTIONS DESIGNED FOR DEBUGGING */
//    std::uint32_t getBar(std::size_t Idx) {
//        std::uint32_t cur_val;
//        cur_val = histogram.at(Idx);
//        return cur_val;
//    }
//
//    std::string getHistogram() {
//        std::string debugStr;
//        for (std::size_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            uint32_t val = histogram.at(i);
//            debugStr += fmt::format("{}, ", val);
//        }
//        return debugStr;
//    }
//
//    std::uint32_t getColumnSum() {
//        std::uint32_t val;
//        for (std::size_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            val += histogram.at(i) * i;
//        }
//        return val;
//    }
//};
//
//#endif

#ifndef HISTOGRAM_TC_HPP
#define HISTOGRAM_TC_HPP

#include "fmt/core.h"
#include "spdlog/spdlog.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

class HistogramTC {
public:
    static const uint32_t HISTOGRAM_LEN = 16;
    uint32_t init_reg_num;
    double alpha;
    std::array<uint32_t, HISTOGRAM_LEN> histogram;
    uint32_t base = 0;

    HistogramTC(uint32_t reg_num) {
        this->init_reg_num = reg_num;

        if (reg_num == 16) {
            alpha = 0.673;
        } else if (reg_num == 32) {
            alpha = 0.697;
        } else if (reg_num == 64) {
            alpha = 0.709;
        } else {
            alpha = 0.7213 / (1 + 1.079 / reg_num);
        }

        histogram = {reg_num, 0};
        base = 0;
    }

    void reset() {
        histogram = {this->init_reg_num, 0};
        base = 0;
    }

    void update(uint32_t cur_val, uint32_t new_val) {
        histogram.at(cur_val)--;
        histogram.at(new_val)++;
    }

    void move(uint32_t cur_val, uint32_t new_val) {
        histogram.at(new_val) = histogram.at(cur_val);
        histogram.at(cur_val) = 0;
    }

    void updateBase(uint32_t delta_B) {
        /* update the value of Base */
        base += delta_B;
        /* update the value of histogram */
        for (uint32_t k = delta_B; k < HISTOGRAM_LEN; ++k) {
            move(k, k - delta_B);
        }
    }

    /* Helper function sigma as defined in
    * "New cardinality estimation algorithms for HyperLogLog sketches"
    * Otmar Ertl, arXiv:1702.01284 */
    double hllSigma(double x) {
        if (x == 1.) return INFINITY;
        double zPrime;
        double y = 1;
        double z = x;
        do {
            x *= x;
            zPrime = z;
            z += x * y;
            y += y;
        } while(zPrime != z);
        return z;
    }

    /* Helper function tau as defined in
     * "New cardinality estimation algorithms for HyperLogLog sketches"
     * Otmar Ertl, arXiv:1702.01284 */
    double hllTau(double x) {
        if (x == 0. || x == 1.) return 0.;
        double zPrime;
        double y = 1.0;
        double z = 1 - x;
        do {
            x = sqrt(x);
            zPrime = z;
            y *= 0.5;
            z -= pow(1 - x, 2)*y;
        } while(zPrime != z);
        return z / 3;
    }

    double getEstimate_new() {
        double z = init_reg_num * hllTau((init_reg_num - histogram.at(HISTOGRAM_LEN - 1)) / (double)init_reg_num);
        for (int j = HISTOGRAM_LEN - 2; j >= 1; --j) {
            z += histogram.at(j);
            z *= 0.5;
        }
        if (base == 0) {
            z += init_reg_num * hllSigma((histogram.at(0)) / (double)init_reg_num);
        }
        else {
            z += histogram.at(0);
            z *= 0.5;
            for(uint32_t j = base - 1; j > 0; --j) {
                z *= 0.5;
            }
        }
        double E = HLL_ALPHA_INF * init_reg_num * init_reg_num / z;
        return E;
    }

    double getEstimate() {
        double power_sum = 0;
        uint32_t bar;
        for (int i = 0; i < HISTOGRAM_LEN; ++i) {
            bar = histogram.at(i);
            int tmp = i + base;
            power_sum += bar * std::pow(2, -tmp);
        }

        double harmonic_avg = 1 / power_sum;
        double card_est = alpha * init_reg_num * init_reg_num * harmonic_avg;

        if (card_est <= 2.5 * init_reg_num) {
            uint32_t v = histogram.at(0);
            if (v != 0) {
                return (double)init_reg_num * log((double)init_reg_num/ v);
            }
        }
        return card_est;
    }

    /* find the minimum value stored in registers using histogram */
    uint32_t getMinBar() {
        uint32_t min_val = 0;
        for (uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
            uint32_t val =  histogram.at(i);
            if (val != 0) {
                min_val = i;
                return min_val;
            }
        }
        return 0;
    }

    /* FUNCTIONS DESIGNED FOR DEBUGGING */
    uint32_t getBar(uint32_t index) {
        uint32_t cur_val = histogram.at(index);
        return cur_val;
    }

    std::string getHistogram() {
        std::string debugStr;
        for (uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
            uint32_t val = histogram.at(i);
            debugStr += fmt::format("{}, ", val);
        }
        return debugStr;
    }

    uint32_t getColumnSum() {
        uint32_t val;
        for (uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
            val += histogram.at(i) * i;
        }
        return val;
    }
};

#endif
