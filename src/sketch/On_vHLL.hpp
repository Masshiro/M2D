#ifndef ON_VHLL_HPP
#define ON_VHLL_HPP


//
//#include "../utils/AccessStatics.hpp"
//#include "../utils/xxhash32.h"
//#include "EMatrix.hpp"
//#include "TwoTupleSketch.h"
//#include <algorithm>
//#include <array>
//#include <cstdint>
//#include <cstdlib>
//#include <memory>
//#include <vector>
//#include <numeric>
//
//class On_vHLL : public TwoTupleSketch {
//public:
//    static const std::uint32_t HISTOGRAM_LEN = 32;
//
//private:
//    std::uint32_t stage_num;
//    std::uint32_t row;
//    std::uint32_t col;
//    std::uint32_t master_seed;
//    std::vector<std::shared_ptr<EMatrix>> stages;
//
//public:
//    Statistics op_statistics;
//
//    On_vHLL(std::uint32_t stage_num, std::uint32_t row, std::uint32_t col, std::uint32_t master_seed)
//            : op_statistics("../../result/sketch/operation_stat/on_vhll.txt") {
//        this->stage_num = stage_num;
//        this->row = row;
//        this->col = col;
//        this->master_seed = master_seed;
//        for (std::uint32_t s = 0; s < stage_num; ++s) {
//            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
//            stages.push_back(std::make_shared<EMatrix>(row, col, stage_seed));
//        }
//    }
//
//    int memory_usage_in_bits() {
//        return 5 * row * col * stage_num;
//    }
//
//    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len,
//                   const void *ptr_element_id, uint64_t element_id_len) {
//        bool FLAG_UPDATED = false;
//        for (auto &stage : stages) {
//            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
//            std::uint32_t select_col = stage->selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
//            std::uint8_t leader_zero = stage->countLeaderZero(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
//            op_statistics.addHashCnt(3);
//
//            std::uint8_t cur_val = stage->estimators[select_row][select_col];
//            op_statistics.addMemoryAccess(1);
//
//            if (cur_val >= HISTOGRAM_LEN) {
//                cur_val = HISTOGRAM_LEN - 1;
//            }
//
//            if (cur_val >= leader_zero) {
//                continue;
//            }
//
//            FLAG_UPDATED = true;
//
//            stage->estimators[select_row][select_col] = leader_zero;
//            op_statistics.addMemoryAccess(1);
//            stage->IUU.at(select_row)->update(cur_val, leader_zero);
//            op_statistics.addMemoryAccess(2);
//            stage->global_histogram.update(cur_val, leader_zero);
//
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
//            stage_res.push_back(stage->IUU.at(select_row)->getEstimate());
//            op_statistics.addMemoryAccess(1);
//            stage_total_spread.push_back(stage->global_histogram.getEstimate());
//        }
//        double n_d_hat = getMedian(stage_res);
//        double n_hat = getMedian(stage_total_spread);
//        double factor = 1;
//        if (row > 1)
//            factor = (double)row / (row - 1);
//        double n_f_hat = factor * (n_d_hat - n_hat / row);
//        return n_f_hat;
//
//        /* Using sum of IUUs to estimate n_hat */
////        std::vector<double> stage_res;
////        std::vector<double> stage_total_spread;
////        for (auto &stage : stages) {
////            double total_card = 0;
////            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
////            stage_res.push_back(stage->IUU.at(select_row)->getEstimate_new());
////
////            for(std::uint32_t i = 0; i < stage->row; i++)
////                total_card += stage->IUU.at(i)->getEstimate_new();
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
//                total_card += stage->IUU.at(i)->getEstimate();
//            stage_total_spread.push_back(total_card);
//        }
//        double n_hat = getAvg(stage_total_spread);
//        return n_hat;
//    }
//
//    double getNhatFast() {
////        std::vector<double> stage_res;
//        std::vector<double> stage_total_spread;
//        for (auto &stage : stages) {
//            stage_total_spread.push_back(stage->global_histogram.getEstimate());
//        }
//        double n_hat = getMedian(stage_total_spread);
//        return n_hat;
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
//                    XXHash32::hash(&master_seed, sizeof(master_seed), s);
//            stages.at(s)->resetSeed(stage_seed);
//        }
//    }
//};


#include "../utils/AccessStatics.hpp"
#include "../utils/xxhash32.h"
#include "EMatrix.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

class On_vHLL : public TwoTupleSketch {
private:
    uint32_t stage_num;
    uint32_t row;
    uint32_t col;
    uint32_t master_seed;
    std::vector<std::shared_ptr<EMatrix>> stages;

public:
    On_vHLL(uint32_t stage_num, uint32_t row, uint32_t col, uint32_t master_seed) {
        this->stage_num = stage_num;
        this->row = row;
        this->col = col;
        this->master_seed = master_seed;
        for (uint32_t s = 0; s < stage_num; ++s) {
            uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.push_back(std::make_shared<EMatrix>(row, col, stage_seed));
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

    double getNHat(bool quick_pick = false) {
        std::vector<double> stage_res;
        for (auto &stage : stages) {
            stage_res.push_back(stage->queryNHat(quick_pick));
        }
        double n_hat = aggregate(stage_res);
        return n_hat;
    }

    static double aggregate(std::vector<double> &stage_res) {
        return getMedian(stage_res);
//        return getAvg(stage_res);
//        return getMin(stage_res);
    }

    void mergeFlow(On_vHLL src_sketch) {
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
        std::sort(v.begin(), v.end());
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
            uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
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
