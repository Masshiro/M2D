#ifndef TON_VHLL_HPP
#define TON_VHLL_HPP

//#include "../utils/AccessStatics.hpp"
//#include "../utils/xxhash32.h"
//#include "EMatrixTC.hpp"
//#include "TwoTupleSketch.h"
//#include <algorithm>
//#include <array>
//#include <cstdint>
//#include <cstdlib>
//#include <memory>
//#include <numeric>
//#include <vector>
//
//class Ton_vHLL : public TwoTupleSketch {
//public:
//    static const std::uint32_t HISTOGRAM_LEN = 16;
//    std::uint32_t stage_num;
//    std::uint32_t row;
//    std::uint32_t col;
//    std::uint32_t master_seed;
//    std::vector<std::shared_ptr<EMatrixTC>> stages;
//    Statistics op_statistics;
//
//    Ton_vHLL(std::uint32_t stage_num, std::uint32_t row, std::uint32_t col, std::uint32_t master_seed)
//        : op_statistics("../../result/sketch/operation_stat/ton_vhll.txt") {
//        this->stage_num = stage_num;
//        this->row = row;
//        this->col = col;
//        this->master_seed = master_seed;
//        for (std::uint32_t s = 0; s < stage_num; ++s) {
//            std::uint32_t stage_seed =
//                XXHash32::hash(&master_seed, sizeof(master_seed), s);
//            stages.push_back(std::make_shared<EMatrixTC>(row, col, stage_seed));
//        }
//    }
//
//    int memory_usage_in_bits() {
//        return 4 * row * col * stage_num;
//    }
//
//    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len,
//                   const void *ptr_element_id, uint64_t element_id_len) {
//        bool FLAG_UPDATED = false;
//        for (auto &stage : stages) {
//            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
//            std::uint32_t select_col = stage->selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
//            std::uint32_t physical_col = select_col / 2;
//            std::uint8_t leader_zero = stage->countLeaderZero(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
//            op_statistics.addHashCnt(3);
//
//            std::uint32_t range = HISTOGRAM_LEN;
//            std::uint8_t new_val;
//
//            /* overflow detected */
//            if (leader_zero >= range + stage->IUU.at(select_row)->base) {
//                /* delta_B refers to the increment of Base */
//                std::uint32_t delta_B = stage->IUU.at(select_row)->getMinBar();
//                /* if delta_B > 0, means Base can be updated */
//                if (delta_B > 0) {
//                    op_statistics.addRebaseCnt(1);
//
//                    /* update values of one specific column in register since their Base has updated */
//                    for (std::uint32_t j = 0; j < col * 2; ++j) {
//                        /* if the column index is even number */
//                        if (j % 2 == 0)
//                            stage->estimators[select_row][j / 2] -= delta_B * 16;
//                        /* if the column index is odd number */
//                        else
//                            stage->estimators[select_row][j / 2] -= delta_B;
//                    }
//                    /* update the value of Base and histogram*/
//                    stage->IUU.at(select_row)->updateBase(delta_B);
//                }
//            }
//
//            std::uint8_t cur_val = stage->estimators[select_row][physical_col];
//            op_statistics.addMemoryAccess(1);
//
//            if (select_col % 2 == 0) {
//                cur_val &= 240; // b'11110000
//                cur_val >>= 4;
//            }
//            else {
//                cur_val &= 15; // b'00001111
//            }
//
//            if (leader_zero < stage->IUU.at(select_row)->base)
//                continue;
//            if (leader_zero - stage->IUU.at(select_row)->base >= range - 1) {
//                new_val = range - 1;
//            }
//            else {
//                new_val = leader_zero - stage->IUU.at(select_row)->base;
//            }
//
//            if (cur_val >= new_val) {
//                continue;
//            }
//
//            FLAG_UPDATED = true;
//
//            uint32_t cur_base = stage->IUU.at(select_row)->base;
//
//            if (select_col % 2 == 0) {
//                new_val <<= 4;
//                stage->estimators[select_row][physical_col] &= 15; // b'00001111
//                stage->estimators[select_row][physical_col] |= new_val;
//                stage->IUU.at(select_row)->update(cur_val, new_val >> 4);
//                stage->global_histogram.update(cur_val + cur_base, (new_val >> 4) + cur_base);
//            }
//            else {
//                stage->estimators[select_row][physical_col] &= 240; // b'11110000
//                stage->estimators[select_row][physical_col] |= new_val;
//                stage->IUU.at(select_row)->update(cur_val, new_val);
//                stage->global_histogram.update(cur_val + cur_base, new_val + cur_base);
//            }
//            op_statistics.addMemoryAccess(3); // 1 for estimator, and 2 for IUU
//        }
//        if (FLAG_UPDATED) {
//            op_statistics.addSketchUpdateCnt(1);
//        }
//        return FLAG_UPDATED;
//    }
//
//    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
//        op_statistics.addSketchQueryCnt(1);
//
//        /* Using Global histogram to estimate n_hat */
//        std::vector<double> stage_res;
//        std::vector<double> stage_total_spread;
//        for (auto &stage : stages) {
//            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
//            op_statistics.addHashCnt(1);
//
//            stage_res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
//            op_statistics.addMemoryAccess(1);
//            stage_total_spread.push_back(stage->global_histogram.getEstimate());
//        }
//        double n_f_hat = getMedian(stage_res);
//        double n_hat = getMedian(stage_total_spread);
//        double factor = 1;
//        if (row > 1)
//            factor = (double)row / (row - 1);
//        return factor * (n_f_hat - n_hat / row);
//
//
//        /* Using sum of IUUs to estimate n_hat */
////        std::vector<double> stage_res;
////        std::vector<double> stage_total_spread;
////        for (auto &stage : stages) {
////            double total_card = 0;
////            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
////            stage_res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
////            for(std::uint32_t i = 0; i < stage->row; i++)
////                total_card += stage->IUU.at(i)->getEstimate(stage->IUU.at(i)->base);
////            stage_total_spread.push_back(total_card);
////        }
////        double n_d_hat = getAvg(stage_res);
////        double n_hat = getAvg(stage_total_spread);
////        double factor = 1;
////        if (row > 1)
////            factor = (double)row / (row - 1);
////        double n_f_hat = factor * (n_d_hat - n_hat / row);
////        return n_f_hat;
//    }
//
//    double getNhatSlow() {
//        std::vector<double> stage_total_spread;
//        for (auto &stage : stages) {
//            double total_card = 0;
//            for(std::uint32_t i = 0; i < stage->row; i++)
//                total_card += stage->IUU.at(i)->getEstimate(stage->IUU.at(i)->base);
//            stage_total_spread.push_back(total_card);
//        }
//        double n_hat = getAvg(stage_total_spread);
//        return n_hat;
//    }
//
//    double getNhatFast() {
//        std::vector<double> stage_total_spread;
//        for (auto &stage : stages) {
//            stage_total_spread.push_back(stage->global_histogram.getEstimate());
//        }
//        double n_hat = getMedian(stage_total_spread);
//        return n_hat;
//    }
//
//    void mergeFlow(Ton_vHLL src_sketch) {
//        for (std::uint32_t l = 0; l < stage_num; ++l) {
//            auto src_stage = src_sketch.stages.at(l);
//            auto dst_stage = stages.at(l);
//            for (std::uint32_t select_row = 0; select_row < row; ++select_row) {
//                std::uint32_t delta_B_src = src_stage->IUU.at(select_row)->getMinBar();
//                std::uint32_t delta_B_dst = dst_stage->IUU.at(select_row)->getMinBar();
//                std::uint32_t newB = std::max(src_stage->IUU.at(select_row)->base + delta_B_src, dst_stage->IUU.at(select_row)->base + delta_B_dst);
//                HistogramTC newQ = {col * 2};
//
//                for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
//                    std::uint32_t physical_col = select_col / 2;
//                    std::uint8_t src_val = src_stage->estimators[select_row][physical_col];
//                    std::uint8_t dst_val = dst_stage->estimators[select_row][physical_col];
//                    if (select_col % 2 == 0) {
//                        src_val &= 240; // b'11110000
//                        src_val >>= 4;
//                        dst_val &= 240;
//                        dst_val >>= 4;
//                    } else {
//                        src_val &= 15; // b'00001111
//                        dst_val &= 15;
//                    }
//
//                    std::uint32_t range = 16;
//
//
//                    std::uint8_t src_sum = src_val + src_stage->IUU.at(select_row)->base;
//                    std::uint8_t dst_sum = dst_val + dst_stage->IUU.at(select_row)->base;
//
//                    dst_stage->global_histogram.histogram.at(src_sum)--;
//
//                    std::uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);
//
//                    if (select_col % 2 == 0) {
//                        dst_stage->estimators[select_row][physical_col] &= 15; // b'00001111
//                        dst_stage->estimators[select_row][physical_col] |= (new_val << 4);
//                    }
//                    else {
//                        dst_stage->estimators[select_row][physical_col] &= 240; // b'11110000
//                        dst_stage->estimators[select_row][physical_col] |= new_val;
//                    }
//
//                    dst_stage->global_histogram.histogram.at(new_val + newB)++;
//                    newQ.update(0, new_val);
//                    dst_stage->IUU.at(select_row)->base = newB;
//
//                    for(std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
//                        dst_stage->IUU.at(select_row)->histogram.at(i) = newQ.histogram.at(i);
//                    }
//                }
//            }
//        }
//    }
//
//    void flowTrace(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
//        bool FLAG = false;
//        FLAG = offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
//        if (FLAG) {
//            decodeFlow(ptr_flow_id, flow_id_len);
//        }
//    }
//
//    inline double getMedian(std::vector<double> &v) {
//        std::uint32_t len = v.size();
//        if (len == 0) {
//            return 0;
//        }
//        std::sort(v.begin(), v.end());
//        double ret = 0;
//        if (len % 2 == 0) {
//            ret = 0.5 * v.at(len / 2) + 0.5 * v.at(len / 2 - 1);
//        } else {
//            ret = v.at(len / 2);
//        }
//        return ret;
//    }
//
//    inline double getAvg(std::vector<double> &v) {
//        std::uint32_t len = v.size();
//        if (len == 0) {
//            return 0;
//        }
//        double sum = accumulate(begin(v), end(v), 0.0);
//        double avg = sum / v.size();
//
//        return avg;
//    }
//
//    void saveStatistic(double pkt_cnt = 1) { op_statistics.save_to_file(pkt_cnt); }
//
//    void resetSketch() {
//        for (auto s : stages) {
//            s->reset();
//        }
//        op_statistics.reset();
//    }
//
//    void resetSeed(uint32_t new_seed) {
//        this->master_seed = new_seed;
//        for (std::uint32_t s = 0; s < stage_num; ++s) {
//            std::uint32_t stage_seed =
//                XXHash32::hash(&master_seed, sizeof(master_seed), s);
//            stages.at(s)->resetSeed(stage_seed);
//        }
//    }
//
//};


#include "../utils/AccessStatics.hpp"
#include "../utils/xxhash32.h"
#include "EMatrixTC.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

class Ton_vHLL : public TwoTupleSketch {
private:
    uint32_t stage_num;
    uint32_t row;
    uint32_t col;
    uint32_t master_seed;
    std::vector<std::shared_ptr<EMatrixTC>> stages;

public:
    Ton_vHLL(uint32_t stage_num, uint32_t row, uint32_t col, uint32_t master_seed) {
        this->stage_num = stage_num;
        this->row = row;
        this->col = col;
        this->master_seed = master_seed;
        for (uint32_t s = 0; s < stage_num; ++s) {
            uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.push_back(std::make_shared<EMatrixTC>(row, col, stage_seed));
        }
    }

    int memory_usage_in_bits() {
        return 4 * row * col * stage_num;
    }

    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len,
                   const void *ptr_element_id, uint64_t element_id_len) override {
        bool FLAG_UPDATED = false;
        for (auto &stage : stages) {
            FLAG_UPDATED |= stage->update(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        }
        return FLAG_UPDATED;
    }

    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) override {
        std::vector<double> stage_res;
        for (auto &stage : stages) {
            stage_res.push_back(stage->query(ptr_flow_id, flow_id_len, true));
        }
        double n_f_hat = aggregate(stage_res);
        return n_f_hat;
    }

    double getNHat() {
        std::vector<double> stage_res;
        for (auto &stage : stages) {
            stage_res.push_back(stage->queryNHat(false));
        }
        double n_hat = aggregate(stage_res);
        return n_hat;
    }

    static double aggregate(std::vector<double> &stage_res) {
        return getMedian(stage_res);
//        return getAvg(stage_res);
//        return getMin(stage_res);
    }

    void mergeFlow(Ton_vHLL src_sketch) {
        for (uint32_t l = 0; l < stage_num; ++l) {
            auto src_stage = src_sketch.stages.at(l);
            auto dst_stage = stages.at(l);
            dst_stage->merge(src_stage);
        }
    }

    void flowTrace(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        decodeFlow(ptr_flow_id, flow_id_len);
    }

    static inline double getMedian(std::vector<double> &v) {
        uint32_t len = v.size();
        if (len == 0) {
            return 0;
        }
        sort(v.begin(), v.end());
        double med;
        if (len % 2 == 0) {
            med = 0.5 * v.at(len / 2) + 0.5 * v.at(len / 2 - 1);
        } else {
            med = v.at(len / 2);
        }
        return med;
    }

    static inline double getAvg(std::vector<double> &v) {
        uint32_t len = v.size();
        if (len == 0) {
            return 0;
        }
        double sum = accumulate(begin(v), end(v), 0.0);
        double avg = sum / len;
        return avg;
    }

    static inline double getMin(std::vector<double> &v) {
        uint32_t len = v.size();
        if (len == 0) {
            return 0;
        }
        double min = *std::min_element(v.begin(), v.end());
        return min;
    }

    void resetSketch() override {
        for (auto &stage : stages) {
            stage->reset();
        }
    }

    void resetSeed(uint32_t new_seed) override {
        this->master_seed = new_seed;
        for (uint32_t s = 0; s < stage_num; ++s) {
            uint32_t stage_seed =
                    XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.at(s)->resetSeed(stage_seed);
        }
    }

    void getHistogram(int true_card) {
        std::vector<double> stage_total_spread_1;
        std::vector<double> stage_total_spread_2;
        for (auto &stage : stages) {

            double total_card = 0;
            for(uint32_t i = 0; i < stage->row; i++) {
                total_card += stage->IUU.at(i)->getEstimate_new();
                //                spdlog::info("{}", stage->IUU.at(i)->printHistogram());
            }
            stage_total_spread_1.push_back(total_card);
            stage_total_spread_2.push_back(stage->global_histogram.getEstimate_new());
            //            spdlog::info("{}", stage->global_histogram.printHistogram());
            //            spdlog::info("sum:{}, glob:{}", total_card, stage->global_histogram.getEstimate_new());
        }
        double n_hat_1 = getAvg(stage_total_spread_1);
        double n_hat_2 = getAvg(stage_total_spread_2);
        spdlog::info("sum:{}, dev:{}, r_dev:{}", n_hat_1, n_hat_1 - true_card, (n_hat_1 - true_card) / true_card);
        spdlog::info("global_histogram:{}, dev:{}, r_dev:{}", n_hat_2, n_hat_2 - true_card, (n_hat_2 - true_card) / true_card);
    }
};

#endif
