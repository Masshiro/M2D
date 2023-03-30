#ifndef EMATRIX_HPP
#define EMATRIX_HPP

#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "../utils/MurmurHash3.h"
#include "Histogram.hpp"
#include "TwoTupleSketch.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

//class EMatrix {
//public:
//    static std::uint32_t FLOW_HASH_SALT;
//    static std::uint32_t ELEMENT_HASH_SALT;
//    static std::uint32_t VALUE_HASH_SALT;
//
//    std::uint32_t flow_hash_seed;
//    std::uint32_t element_hash_seed;
//    std::uint32_t value_hash_seed;
//
//    std::uint32_t col;
//    std::uint32_t row;
//
//    std::uint8_t **estimators;
//    std::vector<std::shared_ptr<Histogram>> IUU;
//    Histogram global_histogram;
//
//    EMatrix(std::uint32_t row, std::uint32_t col, std::uint32_t seed)
//        : global_histogram(row * col) {
//        this->row = row;
//        this->col = col;
//        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
//        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
//        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
//
//        estimators = new std::uint8_t *[row];
//        for (std::uint32_t i = 0; i < row; i++) {
//            estimators[i] = new  std::uint8_t[col]();
//        }
//
//        for (std::uint32_t i = 0; i < row; i++) {
//            IUU.push_back(std::make_shared<Histogram>(col));
//        }
//    }
//
//    ~EMatrix() {
//        for (std::uint32_t i = 0; i < row; i++) {
//            delete[] estimators[i];
//        }
//        delete[] estimators;
//    }
//    void reset() {
//        for (std::uint32_t i = 0; i < row; i++) {
//            for (std::uint32_t j = 0; j < col; j++) {
//                estimators[i][j] = 0;
//            }
//        }
//        for (auto h : IUU) {
//            h->reset();
//        }
//        global_histogram.reset();
//    }
//
//    void resetSeed(uint32_t seed) {
//        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
//        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
//        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
//    }
//
//    std::uint32_t selectRowIdxByFlowID(const void *ptr_flow_id, std::uint64_t flow_id_len) {
//        return XXHash32::hash(ptr_flow_id, flow_id_len, flow_hash_seed) % row;
//    }
//
//    std::uint32_t selectColIdxByElementID(const void *ptr_flow_id, std::uint64_t flow_id_len, const void *ptr_element_id, std::uint64_t element_id_len) {
//        XXHash32 value_hash_fun(element_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(ptr_flow_id, flow_id_len);
//        std::uint32_t hash_value = value_hash_fun.hash();
//        return hash_value % col;
//    }
//
//    std::uint8_t countLeaderZero(const void *ptr_flow_id, std::uint64_t flow_id_len, const void *ptr_element_id, std::uint64_t element_id_len) {
//        XXHash32 value_hash_fun(value_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(ptr_flow_id, flow_id_len);
//        std::uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
//        return hash_value;
//    }
//
//    /* Functions designed for debugging */
//    std::string getEstimatorInfo(std::uint32_t selected_row) {
//        std::string debugStr;
//        for (std::uint32_t i = 0; i < col; ++i) {
//            uint32_t val = estimators[selected_row][i];
//            debugStr += fmt::format("{}, ", val);
//        }
//        return debugStr;
//    }
//
//    void checkGlobalHistogram() {
//        for (int i = 0; i < 32; ++i) {
//            uint32_t tmp = 0;
//            for (std::uint32_t j = 0; j < row; ++j) {
//                tmp += IUU.at(j)->histogram.at(i);
//            }
//            spdlog::info("idx:{}, Sum:{}, Glo:{}", i, tmp, global_histogram.histogram.at(i));
//        }
//    }
//};
//uint32_t EMatrix::FLOW_HASH_SALT = 0xdeedbeef;
//uint32_t EMatrix::ELEMENT_HASH_SALT = 0x123456;
//uint32_t EMatrix::VALUE_HASH_SALT = 0x765432;


#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "Histogram.hpp"
#include "TwoTupleSketch.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

class EMatrix {
public:
    static uint32_t FLOW_HASH_SALT;
    static uint32_t ELEMENT_HASH_SALT;
    static uint32_t VALUE_HASH_SALT;

    uint32_t flow_hash_seed;
    uint32_t element_hash_seed;
    uint32_t value_hash_seed;

    uint32_t col;
    uint32_t row;

    uint8_t **estimators;
    std::vector<std::shared_ptr<Histogram>> IUU;
    Histogram global_histogram;

    static const uint32_t HISTOGRAM_LEN = 32;

    EMatrix(uint32_t row, uint32_t col, uint32_t seed)
            : global_histogram(row * col) {
        this->row = row;
        this->col = col;
        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
        this->value_hash_seed = seed ^ VALUE_HASH_SALT;

        estimators = new uint8_t *[row];
        for (uint32_t i = 0; i < row; i++) {
            estimators[i] = new uint8_t[col]();
        }

        for (uint32_t i = 0; i < row; i++) {
            IUU.push_back(std::make_shared<Histogram>(col));
        }
    }

    ~EMatrix() {
        for (uint32_t i = 0; i < row; i++) {
            delete[] estimators[i];
        }
        delete[] estimators;
    }

    void reset() {
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < col; j++) {
                estimators[i][j] = 0;
            }
        }
        for (auto &h : IUU) {
            h->reset();
        }
        global_histogram.reset();
    }

    void resetSeed(uint32_t seed) {
        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
    }

    uint32_t select_row(const void *ptr_flow_id, uint64_t flow_id_len) {
        return XXHash32::hash(ptr_flow_id, flow_id_len, flow_hash_seed) % row;
    }

    uint32_t select_col(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        XXHash32 value_hash_fun(element_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        uint32_t hash_value = value_hash_fun.hash();
        return hash_value % col;
    }

    bool update(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        uint32_t r_idx = select_row(ptr_flow_id, flow_id_len);
        uint32_t c_idx = select_col(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        uint8_t new_val = count_leading_zeros(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len, value_hash_seed);
        uint8_t cur_val = estimators[r_idx][c_idx];

        if (cur_val >= new_val) {
            return false;
        }

        estimators[r_idx][c_idx] = new_val;
        IUU.at(r_idx)->update(cur_val, new_val);
        global_histogram.update(cur_val, new_val);
        return true;
    }

    double query(const void *ptr_flow_id, uint64_t flow_id_len, bool global_histogram_enabler) {
        uint32_t r_idx = select_row(ptr_flow_id, flow_id_len);
        double n_d_l_hat = IUU.at(r_idx)->getEstimate_new();
        double n_l_hat = 0;

        // estimate n_hat FAST, with global histogram
        if (global_histogram_enabler) {
            n_l_hat = global_histogram.getEstimate_new();
        }
            // estimate n_hat ACCURATELY, with sum of IUUs
        else {
            for(uint32_t i = 0; i < row; i++) {
                n_l_hat += IUU.at(i)->getEstimate_new();
            }
        }

        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_l_hat = factor * (n_d_l_hat - n_l_hat / row);

//        if (n_f_l_hat < 0) {
//            n_f_l_hat = 0;
//        }

        return n_f_l_hat;
    }

    double queryNHat(bool global_histogram_enabler) {
        double n_l_hat = 0;

        // estimate n_hat FAST, with global histogram
        if (global_histogram_enabler) {
            n_l_hat = global_histogram.getEstimate_new();
        }
            // estimate n_hat ACCURATELY, with sum of IUUs
        else {
            for(uint32_t i = 0; i < row; i++) {
                n_l_hat += IUU.at(i)->getEstimate_new();
            }
        }

        return n_l_hat;
    }

    void merge(const std::shared_ptr<EMatrix>& src_stage) {
        for (uint32_t r_idx = 0; r_idx < row; ++r_idx) {
            Histogram newQ = {col};

            for (uint32_t c_idx = 0; c_idx < col; ++c_idx) {
                uint8_t src_val = src_stage->estimators[r_idx][c_idx];
                uint8_t dst_val = estimators[r_idx][c_idx];

                global_histogram.histogram.at(src_val)--;
                uint8_t new_val = std::max(src_val, dst_val);
                estimators[r_idx][c_idx] = new_val;
                global_histogram.histogram.at(new_val)++;
                newQ.update(0, new_val);

                for(uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                    IUU.at(r_idx)->histogram.at(i) = newQ.histogram.at(i);
                }
            }
        }
    }

    /* Functions designed for debugging */
    std::string getEstimatorInfo(uint32_t selected_row) {
        std::string debugStr;
        for (uint32_t i = 0; i < col; ++i) {
            uint32_t val = estimators[selected_row][i];
            debugStr += fmt::format("{}, ", val);
        }
        return debugStr;
    }

    void checkGlobalHistogram() {
        for (int i = 0; i < 32; ++i) {
            uint32_t tmp = 0;
            for (uint32_t j = 0; j < row; ++j) {
                tmp += IUU.at(j)->histogram.at(i);
            }
            spdlog::info("idx:{}, Sum:{}, Glo:{}", i, tmp, global_histogram.histogram.at(i));
        }
    }
};

uint32_t EMatrix::FLOW_HASH_SALT = 0xdeedbeef;
uint32_t EMatrix::ELEMENT_HASH_SALT = 0x123456;
uint32_t EMatrix::VALUE_HASH_SALT = 0x765432;

#endif
