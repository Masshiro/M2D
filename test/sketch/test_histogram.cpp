#include "../src/sketch/Histogram.hpp"
#include "../src/utils/leader_zero.h"
#include "../src/utils/xxhash32.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <random>
#include <vector>
#include <string>

TEST(Histogram, basic) {
    const size_t init_reg = 20;
    Histogram histogram(init_reg);
    ASSERT_EQ(histogram.histogram[0], init_reg);
    histogram.update(0, 1);
    ASSERT_EQ(histogram.histogram[0], init_reg - 1);
    // ASSERT_EQ(histogram.histogram[1], 1);

    histogram.reset();
    ASSERT_EQ(histogram.histogram[0], init_reg);
}

TEST(Histogram, estimate) {
    std::string dat_path =
        "data/caida/equinix-chicago.dirA.20160121-130000.UTC.anon.pcap.dat";
    // double true_result = 905324;
    const size_t init_reg = 20;
    Histogram histogram(init_reg);
    double res = histogram.getEstimate();
    spdlog::info("{}", res);
}

TEST(Histogram, result) {
    std::random_device rd;
    std::uint32_t val_seed = rd();
    std::uint32_t idx_seed = rd();
    std::size_t total_flow = 12500;
    constexpr std::size_t array_len = 512;
    std::uint8_t reg_array[array_len] = {
        0,
    };
    Histogram histogram1(array_len, true);
    Histogram histogram2(array_len, true);

    for (std::size_t i = 0; i < total_flow; ++i) {
        std::uint32_t hashval = XXHash32::hash(&i, sizeof(i), val_seed);
        std::size_t idx = XXHash32::hash(&i, sizeof(i), idx_seed) % array_len;
        std::uint8_t oldval = reg_array[idx];
        std::uint8_t newval = get_leader_zero(hashval);
        if (oldval >= newval) {
            continue;
        }
        histogram1.update(oldval, newval);
        reg_array[idx] = newval;
    }

    for (int i = 0; i < array_len; ++i) {
        std::uint8_t cur = reg_array[i];
        histogram2.update(0, cur);
    }

    double res1 = histogram1.getEstimate();
    double res2 = histogram2.getEstimate();
    std::cout << histogram1.printHistogram() << std::endl;
    std::cout << histogram2.printHistogram() << std::endl;
    ASSERT_EQ(res1, res2);
    spdlog::info("res = {}", res1);
}
