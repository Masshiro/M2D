#include "../../src/utils/leader_zero.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"

TEST(leading_zero, basic) {
    uint64_t random_value = 0x1;
    for (int i = 0; i < 40; i++)
    {
        uint64_t input_value = random_value << i;
        int res = hllPatLen(input_value);
        spdlog::info("Hash string: {:#b}, leading zeroes: {}", input_value, res);
        //        EXPECT_EQ(res, i + 1);
    }
}