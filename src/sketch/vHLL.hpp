#ifndef VHLL_HPP
#define VHLL_HPP

#include "../utils/AccessStatics.hpp"
#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "Histogram.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

class vHLL : public TwoTupleSketch {
public:
    static const std::size_t HISTOGRAM_LEN = 32;
    static const uint32_t MAGIC_NUM = 0x1234567;

private:
    std::size_t estimators_num;
    std::size_t virtual_estimator_num;
    std::uint32_t master_seed;
    std::shared_ptr<uint8_t[]> estimators;
    std::vector<uint32_t> hashfunseeds;
    Histogram global_histogram;

public:
    Statistics op_statistics;
    vHLL(std::size_t stage_num, std::size_t row, std::size_t col, std::uint32_t master_seed)
        : vHLL(stage_num * row * col, stage_num * col, master_seed) {}

    vHLL(std::size_t estimators_num, std::size_t virtual_estimator_num, std::uint32_t master_seed)
        : global_histogram(estimators_num),
          op_statistics("../../result/sketch/operation_stat/vhll.txt") {
        this->estimators_num = estimators_num;
        this->virtual_estimator_num = virtual_estimator_num;
        this->master_seed = master_seed;
        estimators.reset(new uint8_t[estimators_num]());

        for (std::size_t i = 1; i <= virtual_estimator_num; i++) {
            hashfunseeds.push_back(i * MAGIC_NUM ^ master_seed);
        }
    }

    double getTotal() {
        double power_sum = 0;
        double alpha = 0.7213 / (1 + 1.079 / estimators_num);
        for (std::size_t i = 0; i < estimators_num; ++i) {
            int reg = estimators[i];
            power_sum += std::pow(2, -reg);
        }
        power_sum = 1 / power_sum;
        double res = alpha * estimators_num * estimators_num * power_sum;
        // spdlog::debug("m:{}, vm:{}", estimators_num, virtual_estimator_num);
        // spdlog::debug("h: {}, M: {}", res, global_histogram.getEstimate());
        return res;
    }

    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG_UPDATED = false;

        uint32_t select_virtual_hash = XXHash32::hash(ptr_element_id, element_id_len, master_seed) % virtual_estimator_num;
        uint32_t select_idx = XXHash32::hash(ptr_flow_id, flow_id_len,hashfunseeds[select_virtual_hash]) % estimators_num;

        XXHash32 value_hash_fun(master_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        uint8_t new_val = get_leader_zero(value_hash_fun.hash());
        op_statistics.addHashCnt(3);

        uint8_t cur_val = estimators[select_idx];
        op_statistics.addMemoryAccess(1);

        if (new_val >= HISTOGRAM_LEN) {
            new_val = HISTOGRAM_LEN - 1;
        }

        if (cur_val >= new_val) {
            return FLAG_UPDATED;
        }

        FLAG_UPDATED = true;

        estimators[select_idx] = new_val;
        op_statistics.addMemoryAccess(1);
        global_histogram.update(cur_val, new_val);

        op_statistics.addSketchUpdateCnt(1);

        return FLAG_UPDATED;
    }

    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
        op_statistics.addSketchQueryCnt(1);
        std::size_t no_zero_cnt = 0;
        Histogram tmpHistogram(virtual_estimator_num);
        for (auto seed : this->hashfunseeds) {
            uint32_t select_idx = XXHash32::hash(ptr_flow_id, flow_id_len, seed) % estimators_num;
            op_statistics.addHashCnt(1);

            uint8_t tmpval = estimators[select_idx];
            op_statistics.addMemoryAccess(1);

            if (tmpval > 0) {
                no_zero_cnt++;
            }

            if (tmpval >= HISTOGRAM_LEN) {
                tmpval = HISTOGRAM_LEN - 1;
            }

            tmpHistogram.update(0, tmpval);
        }

        double n_s_hat = tmpHistogram.getEstimate();
        double n_hat = global_histogram.getEstimate();

        double factor = ((double)estimators_num * virtual_estimator_num) / (estimators_num - virtual_estimator_num);
        double result = factor * (n_s_hat / virtual_estimator_num - n_hat / estimators_num);
        return result;
    }

    void flowTrace(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG = false;
        FLAG = offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        if (FLAG) {
            decodeFlow(ptr_flow_id, flow_id_len);
        }
    }

    inline double getMedian(std::vector<double> &v) {
        std::size_t len = v.size();
        if (len == 0) {
            return 0;
        }
        std::sort(v.begin(), v.end());
        double ret = 0;
        if (len % 2 == 0) {
            ret = 0.5 * v.at(len / 2) + 0.5 * v.at(len / 2 - 1);
        } else {
            ret = v.at(len / 2);
        }
        return ret;
    }

    void resetSketch() {
        for (std::size_t i = 0; i < estimators_num; ++i) {
            estimators[i] = 0;
        }
        global_histogram.reset();
    }

    void saveStatistic(double pkt_cnt) { op_statistics.save_to_file(pkt_cnt); }

    void resetSeed(uint32_t new_seed) {
        this->master_seed = new_seed;
        hashfunseeds.clear();
        for (size_t i = 1; i <= virtual_estimator_num; i++) {
            hashfunseeds.push_back(i * MAGIC_NUM + master_seed);
        }
    }
};

#endif
