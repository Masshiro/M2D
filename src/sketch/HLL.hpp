#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "spdlog/spdlog.h"
#include "Histogram.hpp"
#include <cstdlib>
#include <vector>
#include <iostream>

class HLL {
private:
    std::vector<std::uint8_t> registers;
    std::uint32_t master_seed;
    Histogram histogram;

public:
    HLL(std::size_t reg_num, std::uint32_t master_seed)
        : master_seed(master_seed), histogram(reg_num) {
        for (std::size_t i = 0; i < reg_num; ++i) {
            registers.push_back(0);
        }
    }

    void offerFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
        std::uint32_t index = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed ^ 0xdeedbeef) % registers.size();
        std::uint32_t hash_value = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed);
        std::uint8_t new_val = get_leader_zero(hash_value);
        std::uint8_t old_val = registers.at(index);
        if (new_val <= old_val)
            return;

        registers.at(index) = new_val;
        histogram.update(old_val, new_val);
    }
    double decodeFlow() { return histogram.getEstimate(); }

    void merge(HLL src_sketch) {
        for (std::uint32_t i = 0; i < registers.size(); ++i) {
            
            std::uint8_t new_val = src_sketch.registers.at(i);
            std::uint8_t old_val = registers.at(i);

            if (src_sketch.registers.at(i) >= registers.at(i)) {
                registers.at(i) = new_val;
                histogram.update(old_val, new_val);
            }
            // spdlog::info("result_val = {}, dst_val = {}, src_val = {}", registers.at(i), dst_sketch.registers.at(i), src_sketch.registers.at(i));
        }
    }

    void resetSketch() {
        for (auto &reg : registers) {
            reg = 0;
        }
        histogram.reset();
    }
    void resetSeed(uint32_t new_seed) { master_seed = new_seed; }
};
