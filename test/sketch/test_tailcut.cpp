#include "../src/sketch/HLL.hpp"
#include "../src/sketch/On_vHLL.hpp"
#include "../src/sketch/Ton_vHLL.hpp"
#include "../src/sketch/vHLL.hpp"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"

#include <ctime>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

TEST(EquivalenceTest, basic) {
    std::random_device rd;
    auto my_seed = rd();
    std::size_t repeat_time = 100;
    std::vector<double> test_num = {1e2, 1e3, 1e4, 1e5};
    spdlog::set_level(spdlog::level::debug);
    std::size_t stage_num = 4, stage_row = 1024, stage_col = 128;
    const std::uint32_t flow_id = 0x1;

    On_vHLL sketch(stage_num, stage_row, stage_col, my_seed);
    Ton_vHLL sketch1(stage_num, stage_row, stage_col/2, my_seed);
    std::vector<double> r_err_vec;
    std::vector<double> rmse_vec;
    std::vector<double> r_err_vec1;
    std::vector<double> rmse_vec1;
    std::vector<double> r_err_vec2;
    std::vector<double> rmse_vec2;
    int test_round = 0;
    for (auto num : test_num) {
        spdlog::info("ROUND {}, TRUE CARD:{}", ++test_round, num);
        r_err_vec.clear();
        rmse_vec.clear();
        r_err_vec1.clear();
        rmse_vec1.clear();
        r_err_vec2.clear();
        rmse_vec2.clear();
        for (std::size_t round = 0; round < repeat_time; ++round) {
            auto new_seed = rd();
            sketch.resetSketch();
            sketch.resetSeed(new_seed);
            sketch1.resetSketch();
            sketch1.resetSeed(new_seed);

            for (int i = 0; i < num; ++i) {
                sketch.offerFlow(&flow_id, sizeof(flow_id), &i, sizeof(i));
                sketch1.offerFlow(&flow_id, sizeof(flow_id), &i, sizeof(i));
            }
            double res = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            double res1 = sketch1.decodeFlow(&flow_id, sizeof(flow_id));
            r_err_vec.push_back(res);
            r_err_vec1.push_back(res1);

            double r_err = (res - num) / num;
            double r_err_pow2 = std::pow(r_err, 2);
            double r_err1 = (res1 - num) / num;
            double r_err_pow21 = std::pow(r_err1, 2);
            rmse_vec.push_back(r_err_pow2);
            rmse_vec1.push_back(r_err_pow21);
        }
        double est_card = 0;
        double est_card1 = 0;
        for (auto v : r_err_vec) {
            est_card += v;
        }
        for (auto v : r_err_vec1) {
            est_card1 += v;
        }

        double r_err_pow2 = 0;
        for (auto v : rmse_vec) {
            r_err_pow2 += v;
        }
        est_card /= repeat_time;
        double r_err = (est_card - num) / num;
        r_err_pow2 /= rmse_vec.size();
        double rmse = std::sqrt(r_err_pow2) / num;
        spdlog::info("On-vHLL\test = {}\tr_err = {}\trmse = {}", est_card, r_err, rmse);

        double r_err_pow21 = 0;
        for (auto v : rmse_vec1) {
            r_err_pow21 += v;
        }
        est_card1 /= repeat_time;
        double r_err1 = (est_card1 - num) / num;
        r_err_pow21 /= rmse_vec1.size();
        double rmse1 = std::sqrt(r_err_pow21) / num;
        spdlog::info("Ton-vHLL\test = {}\tr_err = {}\trmse = {}", est_card1, r_err1, rmse1);
    }
}


