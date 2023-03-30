#include "../src/sketch/On_vHLL.hpp"
#include "../src/sketch/vHLL.hpp"
#include "../src/utils/CardinalityMap.hpp"
#include "../src/utils/ThreadPool.hpp"
#include "../src/utils/xxhash32.h"
#include "fmt/core.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;

std::random_device rd;

TEST(vHll, basic) {
    spdlog::set_level(spdlog::level::debug);
    std::size_t stage_num = 4, stage_row = 1024, stage_col = 128;
    vHLL sketch(stage_num, stage_row, stage_col, rd());
    std::size_t target_flow = 1e4;
    bool insert_noise = true;
    std::size_t noise = 1e6;
    const std::uint32_t flow_id = 0x1;
    for (std::size_t i = 0; i < target_flow; i++) {
        sketch.offerFlow(&flow_id, sizeof(flow_id), &i, sizeof(i));
    }

    if (insert_noise) {
        for (std::size_t i = 0; i < noise - target_flow; i++) {
            std::uint32_t noise_id = rd() | !(0x1);
            std::uint32_t noise_element = rd();
            sketch.offerFlow(&noise_id, sizeof(noise_id), &noise_element,
                             sizeof(noise_element));
        }
    }

    double res = sketch.decodeFlow(&flow_id, sizeof(flow_id));
    spdlog::info(res);
}
