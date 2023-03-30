#include "../../src/sketch/HLL.hpp"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

std::random_device rd;
int my_seed = rd();

TEST(HyperLogLog, basic) {
    std::size_t repeat_time = 100;
    // std::vector<double> test_num = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    std::vector<double> test_num = {1e2, 1e3, 1e4, 1e5};
    std::size_t reg_num = 512;
    HLL sketch(reg_num, rd());
    std::vector<double> r_err_vec;
    std::vector<double> rmse_vec;
    for (auto num : test_num) {
        r_err_vec.clear();
        rmse_vec.clear();
        for (std::size_t round = 0; round < repeat_time; ++round) {
            sketch.resetSketch();
            sketch.resetSeed(rd());
            for (int i = 0; i < num; ++i) {
                sketch.offerFlow(&i, sizeof(i));
            }
            
            double res = sketch.decodeFlow();
            r_err_vec.push_back(res);
            double r_err = (res - num) / num;
            double r_err_pow2 = std::pow(r_err, 2);
            rmse_vec.push_back(r_err_pow2);
        }

        double est_card = 0;
        for (auto v : r_err_vec) {
            est_card += v;
        }

        double r_err_pow2 = 0;
        for (auto v : rmse_vec) {
            r_err_pow2 += v;
        }
        est_card /= repeat_time;
        double r_err = (est_card - num) / num;

        r_err_pow2 /= rmse_vec.size();
        double rmse = std::sqrt(r_err_pow2) / num;

        spdlog::info("est = {}, true = {}, r_err = {}, rmse = {}", est_card,
                     num, r_err, rmse);
    }
}

TEST(HyperLogLog, copy) {
    std::size_t reg_num = 512;
    HLL sketch(reg_num, my_seed);
    sketch.resetSketch();
    for (int i = 0; i < 5000; ++i) {
        sketch.offerFlow(&i, sizeof(i));
    }
    double res = sketch.decodeFlow();
    spdlog::info("est1 = {}, est2 = {}, after assignment = {}, time1 = {}, time2 = {}", res);
}
