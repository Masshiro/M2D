//#ifndef EMATRIX_TC_HPP
//#define EMATRIX_TC_HPP
//
//#include "../utils/leader_zero.h"
//#include "../utils/xxhash32.h"
//#include "Histogram.hpp"
//#include "HistogramTC.hpp"
//#include "PMatrix.hpp"
//#include "TwoTupleSketch.h"
//#include <cstdint>
//#include <cstdlib>
//#include <iostream>
//#include <memory>
//#include <vector>
//
//class EMatrixTC {
//public:
//    static std::uint32_t FLOW_HASH_SALT;
//    static std::uint32_t ELEMENT_HASH_SALT;
//    static std::uint32_t VALUE_HASH_SALT;
//
//    std::size_t col;
//    std::size_t row;
//
//    /* estimators is a matrix of registers */
//    std::uint8_t **estimators;
//    /* IUU(Incremental Update Unit) is a vector, and every element is a shared_ptr,
//     * pointing to an object of the Class of Histogram. */
//    std::vector<std::shared_ptr<HistogramTC>> IUU;
//    Histogram global_histogram;
//
//    std::uint32_t flow_hash_seed;
//    std::uint32_t element_hash_seed;
//    std::uint32_t value_hash_seed;
//
//    EMatrixTC(std::size_t row, std::size_t col, std::size_t seed)
//        : global_histogram(2 * row * col) {
//        this->row = row;
//        this->col = col;
//        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
//        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
//        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
//
//        estimators = new std::uint8_t *[row];
//        for (std::size_t i = 0; i < row; i++) {
//            estimators[i] = new std::uint8_t[col]();
//        }
//
//        for (std::size_t i = 0; i < row; i++) {
//            IUU.push_back(std::make_shared<HistogramTC>(col * 2));
//        }
//    }
//
//    ~EMatrixTC() {
//        for (std::size_t i = 0; i < row; i++) {
//            delete[] estimators[i];
//        }
//        delete[] estimators;
//    }
//    void reset() {
//        for (std::size_t i = 0; i < row; i++) {
//            for (std::size_t j = 0; j < col; j++) {
//                estimators[i][j] = 0;
//            }
//        }
//        for (auto h : IUU) {
//            h->reset();
//        }
//        global_histogram.reset();
//    }
//
//    void reset_row(std::uint32_t r) {
//        for (std::uint32_t j = 0; j < col; j++) {
//            estimators[r][j] = 0;
//        }
//        IUU.at(r)->reset();
////        global_histogram.reset();
//    }
//
//    void resetSeed(uint32_t seed) {
//        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
//        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
//        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
//    }
//
//    std::uint32_t selectRowIdxByFlowID(const void *ptr_flow_id, std::uint64_t flow_id_len) {
//        return XXHash32::hash(ptr_flow_id, flow_id_len,flow_hash_seed) % row;
//    }
//
//    std::uint32_t selectRowIdxByFlowFP(uint32_t key) {
//        return XXHash32::hash(&key, sizeof(key), flow_hash_seed) % row;
//    }
//
//    std::uint32_t selectColIdxByElementID(const void *ptr_flow_id,
//                                        std::uint64_t flow_id_len,
//                                        const void *ptr_element_id,
//                                        std::uint64_t element_id_len) {
//        XXHash32 value_hash_fun(element_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(ptr_flow_id, flow_id_len);
//        std::uint32_t hash_value = value_hash_fun.hash();
//        return hash_value % (col * 2);
//    }
//
//    std::uint32_t selectColIdxByElementFP(uint32_t key,
//                                          const void *ptr_element_id,
//                                          std::uint64_t element_id_len) {
//        XXHash32 value_hash_fun(element_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(&key, sizeof(key));
//        std::uint32_t hash_value = value_hash_fun.hash();
//        return hash_value % (col * 2);
//    }
//
//    std::uint8_t countLeaderZero(const void *ptr_flow_id,
//                                 std::uint64_t flow_id_len,
//                                 const void *ptr_element_id,
//                                 std::uint64_t element_id_len) {
//
//        XXHash32 value_hash_fun(value_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(ptr_flow_id, flow_id_len);
//        // 返回值hash_value为uint8_t(char)类型，1字节长度，8bit
//        std::uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
//        return hash_value;
//    }
//
//    std::uint8_t countLeaderZeroFP(uint32_t key,
//                                 const void *ptr_element_id,
//                                 std::uint64_t element_id_len) {
//
//        XXHash32 value_hash_fun(value_hash_seed);
//        value_hash_fun.add(ptr_element_id, element_id_len);
//        value_hash_fun.add(&key, sizeof(key));
//        // 返回值hash_value为uint8_t(char)类型，1字节长度，8bit
//        std::uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
//        return hash_value;
//    }
//
//    /* FOLLOWING FUNCTIONS ARE DESIGNED FOR DEBUGGING */
//    std::string getBaseInfo() {
//        std::string debugStr;
//        for (std::size_t i = 0; i < row; ++i) {
//            uint32_t val = IUU.at(i)->base;
//            debugStr += fmt::format("{}, ", val);
//        }
//        return debugStr;
//    }
//
//    std::string getEstimatorInfo(std::size_t selected_row) {
//        std::string debugStr;
//        for (std::size_t i = 0; i < col * 2; ++i) {
//            uint32_t val = estimators[selected_row][i];
//            if (i % 2 == 0) {
//                val &= 240; // b'11110000
//                val >>= 4;
//            } else {
//                val &= 15; // b'00001111
//            }
//
//            debugStr += fmt::format("{}, ", val);
//        }
//        return debugStr;
//    }
//};
//
//uint32_t EMatrixTC::FLOW_HASH_SALT = 0xdeedbeef;
//uint32_t EMatrixTC::ELEMENT_HASH_SALT = 0x123456;
//uint32_t EMatrixTC::VALUE_HASH_SALT = 0x765432;
//
//#endif


#ifndef EMATRIX_TC_HPP
#define EMATRIX_TC_HPP

#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "Histogram.hpp"
#include "HistogramTC.hpp"
#include "PMatrix.hpp"
#include "TwoTupleSketch.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

class EMatrixTC {
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
    std::vector<std::shared_ptr<HistogramTC>> IUU;
    Histogram global_histogram;

    static const uint32_t HISTOGRAM_LEN = 16;

    EMatrixTC(uint32_t row, uint32_t col, uint32_t seed)
            : global_histogram(2 * row * col) {
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
            IUU.push_back(std::make_shared<HistogramTC>(col * 2));
        }
    }

    ~EMatrixTC() {
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

    void reset_row(uint32_t r) {
        for (uint32_t j = 0; j < col; j++) {
            estimators[r][j] = 0;
        }
        IUU.at(r)->reset();
//        global_histogram.reset();
    }

    void resetSeed(uint32_t seed) {
        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
    }

    uint32_t selectRowIdxByFlowID(const void *ptr_flow_id, uint64_t flow_id_len) {
        return XXHash32::hash(ptr_flow_id, flow_id_len,flow_hash_seed) % row;
    }

    uint32_t selectRowIdxByFlowFP(uint32_t key) {
        return XXHash32::hash(&key, sizeof(key), flow_hash_seed) % row;
    }

    uint32_t selectColIdxByElementID(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        XXHash32 value_hash_fun(element_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        uint32_t hash_value = value_hash_fun.hash();
        return hash_value % (col * 2);
    }

    uint32_t selectColIdxByElementFP(uint32_t key, const void *ptr_element_id, uint64_t element_id_len) {
        XXHash32 value_hash_fun(element_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(&key, sizeof(key));
        uint32_t hash_value = value_hash_fun.hash();
        return hash_value % (col * 2);
    }

    uint8_t countLeaderZero(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        XXHash32 value_hash_fun(value_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
        return hash_value;
    }

    uint8_t countLeaderZeroFP(uint32_t key, const void *ptr_element_id, uint64_t element_id_len) {
        XXHash32 value_hash_fun(value_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(&key, sizeof(key));
        uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
        return hash_value;
    }

    uint8_t check_overflow(uint8_t leading_zeros, uint32_t cur_val, uint32_t r_idx) {
        uint8_t new_val;
        if (leading_zeros >= HISTOGRAM_LEN + IUU.at(r_idx)->base) {
            uint32_t delta_B = IUU.at(r_idx)->getMinBar();
            if (delta_B > 0) {
                for (uint32_t j = 0; j < col * 2; ++j) {
                    if (j % 2 == 0)
                        estimators[r_idx][j / 2] -= delta_B * 16;
                    else
                        estimators[r_idx][j / 2] -= delta_B;
                }
                IUU.at(r_idx)->updateBase(delta_B);
            }
        }
        if (leading_zeros < IUU.at(r_idx)->base)
            new_val = cur_val;
        else if (leading_zeros - IUU.at(r_idx)->base >= HISTOGRAM_LEN - 1) {
            new_val = HISTOGRAM_LEN - 1;
        }
        else {
            new_val = leading_zeros - IUU.at(r_idx)->base;
        }
        return new_val;
    }

    bool update(uint32_t key, const void *ptr_element_id, uint64_t element_id_len) {
        uint32_t r_idx = selectRowIdxByFlowFP(key);
        uint32_t c_idx = selectColIdxByElementFP(key, ptr_element_id, element_id_len);
        uint8_t leading_zeros = countLeaderZeroFP(key, ptr_element_id, element_id_len);
        uint8_t cur_val = estimators[r_idx][c_idx / 2];
        if (c_idx % 2 == 0) {
            cur_val &= 240; // b'11110000
            cur_val >>= 4;
        }
        else {
            cur_val &= 15; // b'00001111
        }

        uint8_t new_val = check_overflow(leading_zeros, cur_val, r_idx);
        if (cur_val >= new_val) {
            return false;
        }

        uint32_t cur_base = IUU.at(r_idx)->base;

        if (c_idx % 2 == 0) {
            estimators[r_idx][c_idx / 2] &= 15; // b'00001111
            estimators[r_idx][c_idx / 2] |= (new_val << 4);
        }
        else {
            estimators[r_idx][c_idx / 2] &= 240; // b'11110000
            estimators[r_idx][c_idx / 2] |= new_val;
        }
        IUU.at(r_idx)->update(cur_val, new_val);
        global_histogram.update(cur_val + cur_base, new_val + cur_base);
        return true;
    }

    bool update(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        uint32_t r_idx = selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
        uint32_t c_idx = selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        uint8_t leading_zeros = count_leading_zeros(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len, value_hash_seed);
        uint8_t cur_val = estimators[r_idx][c_idx / 2];
        if (c_idx % 2 == 0) {
            cur_val &= 240; // b'11110000
            cur_val >>= 4;
        }
        else {
            cur_val &= 15; // b'00001111
        }

        uint8_t new_val = check_overflow(leading_zeros, cur_val, r_idx);
        if (cur_val >= new_val) {
            return false;
        }

        uint32_t cur_base = IUU.at(r_idx)->base;

        if (c_idx % 2 == 0) {
            estimators[r_idx][c_idx / 2] &= 15; // b'00001111
            estimators[r_idx][c_idx / 2] |= (new_val << 4);
        }
        else {
            estimators[r_idx][c_idx / 2] &= 240; // b'11110000
            estimators[r_idx][c_idx / 2] |= new_val;
        }
        IUU.at(r_idx)->update(cur_val, new_val);
        global_histogram.update(cur_val + cur_base, new_val + cur_base);
        return true;
    }

    double query(uint32_t key, bool global_histogram_enabler) {
        uint32_t r_idx = selectRowIdxByFlowFP(key);
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

    double query(const void *ptr_flow_id, uint64_t flow_id_len, bool global_histogram_enabler) {
        uint32_t r_idx = selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
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

    void merge(const std::shared_ptr<EMatrixTC>& src_stage) {
        for (uint32_t select_row = 0; select_row < row; ++select_row) {
            uint32_t delta_B_src = src_stage->IUU.at(select_row)->getMinBar();
            uint32_t delta_B_dst = IUU.at(select_row)->getMinBar();
            uint32_t newB = std::max(src_stage->IUU.at(select_row)->base + delta_B_src, IUU.at(select_row)->base + delta_B_dst);
            HistogramTC newQ = {col * 2};

            for (uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                uint32_t physical_col = select_col / 2;
                uint8_t src_val = src_stage->estimators[select_row][physical_col];
                uint8_t dst_val = estimators[select_row][physical_col];
                if (select_col % 2 == 0) {
                    src_val &= 240; // b'11110000
                    src_val >>= 4;
                    dst_val &= 240;
                    dst_val >>= 4;
                } else {
                    src_val &= 15; // b'00001111
                    dst_val &= 15;
                }

                uint32_t range = 16;


                uint8_t src_sum = src_val + src_stage->IUU.at(select_row)->base;
                uint8_t dst_sum = dst_val + IUU.at(select_row)->base;

                global_histogram.histogram.at(src_sum)--;

                uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);

                if (select_col % 2 == 0) {
                    estimators[select_row][physical_col] &= 15; // b'00001111
                    estimators[select_row][physical_col] |= (new_val << 4);
                }
                else {
                    estimators[select_row][physical_col] &= 240; // b'11110000
                    estimators[select_row][physical_col] |= new_val;
                }

                global_histogram.histogram.at(new_val + newB)++;
                newQ.update(0, new_val);
                IUU.at(select_row)->base = newB;

                for(uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                    IUU.at(select_row)->histogram.at(i) = newQ.histogram.at(i);
                }
            }
        }
    }

    /* Functions designed for debugging */
    std::string getBaseInfo() {
        std::string debugStr;
        for (uint32_t i = 0; i < row; ++i) {
            uint32_t val = IUU.at(i)->base;
            debugStr += fmt::format("{}, ", val);
        }
        return debugStr;
    }

    std::string getEstimatorInfo(uint32_t selected_row) {
        std::string debugStr;
        for (uint32_t i = 0; i < col * 2; ++i) {
            uint32_t val = estimators[selected_row][i];
            if (i % 2 == 0) {
                val &= 240; // b'11110000
                val >>= 4;
            } else {
                val &= 15; // b'00001111
            }

            debugStr += fmt::format("{}, ", val);
        }
        return debugStr;
    }
};

uint32_t EMatrixTC::FLOW_HASH_SALT = 0xdeedbeef;
uint32_t EMatrixTC::ELEMENT_HASH_SALT = 0x123456;
uint32_t EMatrixTC::VALUE_HASH_SALT = 0x765432;

#endif
