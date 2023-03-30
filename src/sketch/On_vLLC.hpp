#ifndef ON_VLLC_HPP
#define ON_VLLC_HPP

#include "../utils/AccessStatics.hpp"
#include "../utils/xxhash32.h"
#include "EMatrixLL.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <array>
#include <bits/stdint-uintn.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>
#include <numeric>

#define ln2_n 710 // numerator of ln2
#define ln2_d 1024 // denominator of ln2
#define sqrt2_n 14142 // numerator of sqrt2
#define sqrt2_d 10000 // denominator of sqrt2
#define alpha_n 397 // numerator of alpha
#define alpha_d 1000 // denominator of alpha

class On_vLLC : public TwoTupleSketch{
public:
    static const std::uint32_t HISTOGRAM_LEN = 32;
    static constexpr double alpha = 0.39701;
    Statistics op_statistics;

private:
    std::uint32_t stage_num;
    std::uint32_t row;
    std::uint32_t col;
    std::uint32_t master_seed;
    std::vector<std::shared_ptr<EMatrixLL>> stages;

public:
    On_vLLC(std::uint32_t stage_num, std::uint32_t row, std::uint32_t col, std::uint32_t master_seed)
        : op_statistics("../../result/sketch/operation_stat/on_vllc.txt"){
        this->stage_num = stage_num;
        this->row = row;
        this->col = col;
        this->master_seed = master_seed;
        for (std::uint32_t s = 0; s < stage_num; ++s) {
            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.push_back(std::make_shared<EMatrixLL>(row, col, stage_seed));
        }
    }


    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len,
                   const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG_UPDATED = false;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            std::uint32_t select_col = stage->selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
            std::uint8_t leader_zero = stage->countLeaderZero(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
            op_statistics.addHashCnt(3);

            std::uint8_t cur_val = stage->estimators[select_row][select_col];
            op_statistics.addMemoryAccess(1);

            if (cur_val >= HISTOGRAM_LEN) {
                cur_val = HISTOGRAM_LEN - 1;
            }

            if (cur_val >= leader_zero) {
                continue;
            }

            FLAG_UPDATED = true;

            stage->estimators[select_row][select_col] = leader_zero;
            stage->IUU.at(select_row) += leader_zero - cur_val;
            stage->global_reg_sum += leader_zero - cur_val;
            if (cur_val == 0) {
                stage->zero_counter.at(select_row)--;
            }
            op_statistics.addMemoryAccess(2);

        }
        if (FLAG_UPDATED) {
            op_statistics.addSketchUpdateCnt(1);
        }
        return FLAG_UPDATED;
    }

    double getEstimate(std::uint32_t val, std::uint32_t zeros) {
        /* floating point */
//        double row_sum_avg = val / (double)col;
//        double card_est = alpha * col * std::pow(2, row_sum_avg);


        /* integer */
        double card_est = alpha * taylorPowerApprox(val, col);

        if (2 * card_est < 5 * col) {
            uint32_t v = zeros;
            if (v != 0)
                return  (double)col * log((double)col/ v);
        }

        return card_est;
    }

    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
        op_statistics.addSketchQueryCnt(1);
        std::vector<double> stage_res;
        std::vector<double> stage_total_spread;

        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            op_statistics.addHashCnt(1);

            double n_d = getEstimate(stage->IUU.at(select_row), stage->zero_counter.at(select_row));
            op_statistics.addMemoryAccess(1);
            stage_res.push_back(n_d);

            double total_card = 0;

            /* Using sum of IUUs to estimate n_hat */
            for(std::uint32_t i = 0; i < row; ++i) {
                total_card += getEstimate(stage->IUU.at(i), stage->zero_counter.at(i));
            }

            /* Using global register to estimate n_hat(floating point) */
//            double total_sum_avg = stage->global_reg_sum / (double)(col * row);
//            total_card = alpha * row * col * std::pow(2, total_sum_avg);

            /* Using global register to estimate n_hat(integer) */
//            total_card = alpha * powerApprox(stage->global_reg_sum, (col * row));

            stage_total_spread.push_back(total_card);

        }

        double n_d_hat = getAvg(stage_res);
        double n_hat = getAvg(stage_total_spread);
        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_hat = factor * (n_d_hat - n_hat / row);
        return n_f_hat;
    }

    double getMedian(std::vector<double> &v) {
        std::uint32_t len = v.size();
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

    std::uint32_t getMedian(std::vector<std::uint32_t> &v) {
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

    double getAvg(std::vector<double> &v) {
        std::uint32_t len = v.size();
        if (len == 0) {
            return 0;
        }
        double sum = accumulate(begin(v), end(v), 0.0);
        double avg = sum / v.size();

        return avg;
    }

    int powerApprox(int x, int c) {
        int q = x / c;
        int r = x % c;
        return ((1 << q) * ((c + r + (r * (r - c)) / (2 * c))));
    }

    long long taylorPowerApprox(long long x, long long c) {
        long long q = x / c; // quotient
        long long r = x % c; // remainder
        long long res;

        if (2 * r < c) {
            res = (1 << q) * ((c + (ln2_n * r / ln2_d) + (ln2_n * ln2_n * r * r) / (2 * c * ln2_d * ln2_d)));
        } else {
            res = (sqrt2_n * (1 << q) *  (c + (ln2_n * (2 * r - c) / (2 * ln2_d)) + (ln2_n * ln2_n * (2 * r - c) * (2 * r - c) / (8 * c * ln2_d * ln2_d)))) / sqrt2_d;
        }

        return res;
    }

    void flowTrace(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG = false;
        FLAG = offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        if (FLAG) {
            decodeFlow(ptr_flow_id, flow_id_len);
        }
    }

    void resetSketch() {
        for (auto s : stages) {
            s->reset();
        }
        op_statistics.reset();
    }

    void saveStatistic(int pkt_cnt = 1) { op_statistics.save_to_file(pkt_cnt); }

    void resetSeed(uint32_t new_seed) {
        this->master_seed = new_seed;
        for (std::uint32_t s = 0; s < stage_num; ++s) {
            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.at(s)->resetSeed(stage_seed);
        }
    }
};

#endif
