//#ifndef HISTOGRAM_HPP
//#define HISTOGRAM_HPP
//
//#include "fmt/core.h"
//#include "spdlog/spdlog.h"
//#include <array>
//#include <cmath>
//#include <cstdint>
//#include <cstdlib>
//#include <string>
//
//#define HLL_ALPHA_INF 0.721347520444481703680 /* constant for 0.5/ln(2) */
//
//class Histogram {
//public:
//    static const std::uint32_t HISTOGRAM_LEN = 32;
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
//            double ret = std::pow((1 - 1.0 / num_of_estimators), estimation) *
//                         std::log2(1 - 1.0 / num_of_estimators);
//            return ret;
//        }
//
//        double a = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << k));
//        double b = 1 - 1.0 / ((long long)num_of_estimators * (long long)(1 << (k - 1)));
//        double ret = std::pow(a, estimation) * std::log2(a) -
//                     std::pow(b, estimation) * std::log2(b);
//        return ret;
//    }
//
//    static double derivation_of_log_L(double estimation, std::array<uint32_t, HISTOGRAM_LEN> histogram, uint32_t histogram_len, int num_of_estimators) {
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
//        }
//        return ret;
//    }
//
//    double MLE(double card_est) {
//        std::uint32_t bar = 0;
//        std::uint32_t smallest_value = 0;
//        for (std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
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
//    bool lc_mle_enabler;
//    std::array<std::uint32_t, HISTOGRAM_LEN> histogram;
//
//    Histogram(std::uint32_t reg_num, bool lc_mle_enabler = true)
//        : lc_mle_enabler(lc_mle_enabler) {
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
//    }
//
//    void reset() { histogram = {this->init_reg_num, 0}; }
//
//    void update(std::uint32_t oldIdx, std::uint32_t newIdx) {
//        if (oldIdx > HISTOGRAM_LEN || oldIdx > newIdx) {
//            spdlog::error("access histogram: oldIdx:{}, newIdx:{}", oldIdx, newIdx);
//        }
//        if (newIdx > HISTOGRAM_LEN) {
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
//    void move(std::uint32_t oldIdx, std::uint32_t newIdx) {
//        std::uint32_t cur_val;
//        cur_val = histogram.at(oldIdx);
//        histogram.at(oldIdx) = 0;
//        histogram.at(newIdx) = cur_val;
//    }
//
//
//    /* Helper function sigma as defined in
//    * "New cardinality estimation algorithms for HyperLogLog sketches"
//    * Otmar Ertl, arXiv:1702.01284 */
//    double hllSigma(double x) {
//        if (x == 1.) return INFINITY;
//        double zPrime;
//        double y = 1;
//        double z = x;
//        do {
//            x *= x;
//            zPrime = z;
//            z += x * y;
//            y += y;
//        } while(zPrime != z);
//        return z;
//    }
//
//    /* Helper function tau as defined in
//     * "New cardinality estimation algorithms for HyperLogLog sketches"
//     * Otmar Ertl, arXiv:1702.01284 */
//    double hllTau(double x) {
//        if (x == 0. || x == 1.) return 0.;
//        double zPrime;
//        double y = 1.0;
//        double z = 1 - x;
//        do {
//            x = sqrt(x);
//            zPrime = z;
//            y *= 0.5;
//            z -= pow(1 - x, 2)*y;
//        } while(zPrime != z);
//        return z / 3;
//    }
//
//    double getEstimate_new() {
//        double z = init_reg_num * hllTau((init_reg_num - histogram.at(HISTOGRAM_LEN-1)) / (double)init_reg_num);
//        for (int j = HISTOGRAM_LEN - 1; j >= 1; --j) {
//            z += histogram.at(j);
//            z *= 0.5;
//        }
//        z += init_reg_num * hllSigma(histogram.at(0) / (double)init_reg_num);
//        double E = HLL_ALPHA_INF * init_reg_num * init_reg_num / z;
//        return E;
//    }
//
//    double getEstimate() {
//        double power_sum = 0;
//        std::uint32_t bar;
//        for (int i = 0; i < HISTOGRAM_LEN; ++i) {
//            bar = histogram.at(i);
//            int tmp = i;
//            power_sum += bar * std::pow(2, -tmp);
//        }
//        double harmonic_avg = 1 / power_sum;
//        double card_est = alpha * init_reg_num * init_reg_num * harmonic_avg;
//
//        if (!lc_mle_enabler) {
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
//        else if (card_est > 5 * init_reg_num) {
//            return card_est;
//        }
//        return MLE(card_est);
//    }
//
//    /* FUNCTIONS DESIGNED FOR DEBUGGING */
//    std::string printHistogram() {
//        std::string debugStr;
//        for (std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
//            uint32_t val = histogram.at(i);
//            debugStr += fmt::format("{}, ", val);
//        }
//        return debugStr;
//    }
//
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
//    std::uint32_t getMaxBar() {
//        std::uint32_t max_val = HISTOGRAM_LEN - 1;
//        for (int i = HISTOGRAM_LEN - 1; i >= 0; --i) {
//            uint32_t val =  histogram.at(i);
//            if (val != 0) {
//                max_val = i;
//                return max_val;
//            }
//        }
//        return (HISTOGRAM_LEN - 1);
//    }
//};
//
//#endif

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include "fmt/core.h"
#include "spdlog/spdlog.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#define HLL_ALPHA_INF 0.721347520444481703680 /* constant for 0.5/ln(2) */

class Histogram {
public:
    static const uint32_t HISTOGRAM_LEN = 32;
    uint32_t init_reg_num;
    double alpha;
    std::array<uint32_t, HISTOGRAM_LEN> histogram;

    Histogram(uint32_t reg_num) {
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
    }

    void reset() { histogram = {this->init_reg_num, 0}; }

    void update(uint32_t cur_val, uint32_t new_val) {
        histogram.at(cur_val)--;
        histogram.at(new_val)++;
    }

    void move(uint32_t cur_val, uint32_t new_val) {
        histogram.at(new_val) = histogram.at(cur_val);
        histogram.at(cur_val) = 0;
    }

    double MLE(std::array<uint32_t, HISTOGRAM_LEN> c, int q) {
        uint32_t m = init_reg_num;
        const double eps = 1e-2;
        double relativeErrorLimit = eps/(sqrt(m));

        if (c[q+1] == m) return std::numeric_limits<double>::infinity();

        int kMin, kMax;

        for(kMin=0; c[kMin]==0; ++kMin);
        int kMinPrime = std::max(1, kMin);

        for(kMax=q+1; c[kMax]==0; --kMax);
        int kMaxPrime = std::min(q, kMax);

        double z = 0.;
        for (int k = kMaxPrime; k >= kMinPrime; --k) {
            z = 0.5*z + c[k];
        }
        z = ldexp(z, -kMinPrime);

        uint32_t cPrime = c[q+1];
        if (q >= 1) {
            cPrime += c[kMaxPrime];
        }

        double gPrevious = 0;

        double x;
        double a = z + c[0];
        uint32_t mPrime = m - c[0];
        {
            double b = z + ldexp(c[q+1], -q);
            if (b <= 1.5*a) {
                x = mPrime/(0.5*b+a);
            }
            else {
                x = (mPrime/b)*log1p(b/a);
            }
        }

        double deltaX = x;
        while(deltaX > x*relativeErrorLimit) {
            int kappaMinus1;
            frexp(x, &kappaMinus1);
            double xPrime = ldexp(x, -std::max(kMaxPrime+1, kappaMinus1+2));
            double xPrime2 = xPrime*xPrime;
            double h = xPrime - xPrime2/3 + (xPrime2*xPrime2)*(1./45. - xPrime2/472.5);
            for (int k = kappaMinus1; k >= kMaxPrime; --k) {
                double hPrime = 1. - h;
                h = (xPrime + h*hPrime)/(xPrime+hPrime);
                xPrime += xPrime;
            }
            double g = cPrime*h;
            for (int k = kMaxPrime-1; k >= kMinPrime; --k) {
                double hPrime = 1. - h;
                h = (xPrime + h*hPrime)/(xPrime+hPrime);
                xPrime += xPrime;
                g += c[k] * h;
            }
            g += x*a;
            if (gPrevious < g && g <= mPrime) {
                deltaX *= (g-mPrime)/(gPrevious-g);
            }
            else {
                deltaX = 0;
            }
            x += deltaX;
            gPrevious = g;
        }
        return x*m;
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
        z += init_reg_num * hllSigma(histogram.at(0) / (double)init_reg_num);
        double E = HLL_ALPHA_INF * init_reg_num * init_reg_num / z;
        return E;
    }

    double getEstimate() {
        double power_sum = 0;
        uint32_t bar;
        for (int i = 0; i < HISTOGRAM_LEN; ++i) {
            bar = histogram.at(i);
            power_sum += bar * std::pow(2, -i);
        }
        double harmonic_avg = 1 / power_sum;
        double card_est = alpha * init_reg_num * init_reg_num * harmonic_avg;

        if (card_est <= 2 * init_reg_num) {
            uint32_t v = histogram.at(0);
            if (v != 0) {
                return (double)init_reg_num * log((double)init_reg_num/ v);
            }
        }
        if (card_est > 5 * init_reg_num) {
            return card_est;
        }
        return MLE(histogram, HISTOGRAM_LEN - 2);
    }

    /* FUNCTIONS DESIGNED FOR DEBUGGING */
    std::string printHistogram() {
        std::string debugStr;
        for (uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
            uint32_t val = histogram.at(i);
            debugStr += fmt::format("{}, ", val);
        }
        return debugStr;
    }

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

    uint32_t getMaxBar() {
        uint32_t max_val = HISTOGRAM_LEN - 1;
        for (int i = HISTOGRAM_LEN - 1; i >= 0; --i) {
            uint32_t val =  histogram.at(i);
            if (val != 0) {
                max_val = i;
                return max_val;
            }
        }
        return (HISTOGRAM_LEN - 1);
    }
};

#endif
