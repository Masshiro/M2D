#include "../../src/sketch/HLL.hpp"
#include "../../src/sketch/On_vHLL.hpp"
#include "../../src/sketch/Ton_vHLL.hpp"
#include "../../src/sketch/vHLL.hpp"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"

#include <ctime>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

TEST(Distribution, basic) {
    std::random_device rd;
    auto my_seed = rd();
    spdlog::set_level(spdlog::level::debug);
    std::size_t stage_num = 4, stage_row = 1024, stage_col = 8192;
    const std::uint32_t flow_id = 0x1;
    int repeat_time = 1;
    int num = 1024 * stage_col;

    Ton_vHLL sketch(stage_num, stage_row/2, stage_col, my_seed);
    std::vector<double> r_err_vec;
    std::vector<double> rmse_vec;

    r_err_vec.clear();
    rmse_vec.clear();
    for (std::size_t round = 0; round < repeat_time; ++round) {
        auto new_seed = rd();
        sketch.resetSketch();
        sketch.resetSeed(new_seed);

        for (int i = 0; i < num; ++i) {
            sketch.offerFlow(&flow_id, sizeof(flow_id), &i, sizeof(i));
        }
        double res = sketch.decodeFlow(&flow_id, sizeof(flow_id));
        r_err_vec.push_back(res);

        double r_err = (res - num) / num;
        double r_err_pow2 = std::pow(r_err, 2);
    }
    double est_card = 0;
    for (auto v : r_err_vec) {
        est_card += v;
    }

    spdlog::info("Ton-vHLL\test = {}", est_card);


}