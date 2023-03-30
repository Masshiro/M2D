#ifndef PREFILTER_HPP
#define PREFILTER_HPP

#include "../utils/leader_zero.h"
#include "../utils/xxhash32.h"
#include "HistogramTC.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

class PMatrix {
public:
    static std::uint32_t FLOW_HASH_SALT;
    static std::uint32_t ELEMENT_HASH_SALT;
    static std::uint32_t VALUE_HASH_SALT;

    std::uint32_t col;
    std::uint32_t row;

    std::uint8_t **estimators;
    std::vector<std::shared_ptr<HistogramTC>> IUU;
    std::vector<double> global_card;

    std::uint32_t flow_hash_seed;
    std::uint32_t element_hash_seed;
    std::uint32_t value_hash_seed;

    PMatrix(std::uint32_t row, std::uint32_t col, std::uint32_t seed) {
        this->row = row;
        this->col = col;
        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
        this->value_hash_seed = seed ^ VALUE_HASH_SALT;

        estimators = new std::uint8_t *[row];
        for (std::uint32_t i = 0; i < row; i++) {
            estimators[i] = new std::uint8_t[col]();
        }

        for (std::uint32_t i = 0; i < row; i++) {
            IUU.push_back(std::make_shared<HistogramTC>(col * 2));
            global_card.push_back(-1);
        }
    }

    ~PMatrix() {
        for (std::uint32_t i = 0; i < row; i++) {
            delete[] estimators[i];
        }
        delete[] estimators;
    }

    void reset() {
        for (std::uint32_t i = 0; i < row; i++) {
            for (std::uint32_t j = 0; j < col; j++) {
                estimators[i][j] = 0;
            }
            IUU.at(i)->reset();
            global_card.at(i) = -1;
        }
        for (auto h : IUU) {
            h->reset();
        }
    }

    void resetSeed(uint32_t seed) {
        this->flow_hash_seed = seed ^ FLOW_HASH_SALT;
        this->element_hash_seed = seed ^ ELEMENT_HASH_SALT;
        this->value_hash_seed = seed ^ VALUE_HASH_SALT;
    }

    void reset_row(std::uint32_t r) {
        for (std::uint32_t j = 0; j < col; j++) {
            estimators[r][j] = 0;
        }
        IUU.at(r)->reset();
        global_card.at(r) = -1;
    }

    std::uint32_t selectColIdxByElementID(const void *ptr_flow_id,
                                          std::uint64_t flow_id_len,
                                          const void *ptr_element_id,
                                          std::uint64_t element_id_len) {
        XXHash32 value_hash_fun(element_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        std::uint32_t hash_value = value_hash_fun.hash();
        return hash_value % (col * 2);
    }

    std::uint32_t selectColIdxByElementFP(uint32_t key,
                                          const void *ptr_element_id,
                                          std::uint64_t element_id_len) {
        XXHash32 value_hash_fun(element_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(&key, sizeof(key));
        std::uint32_t hash_value = value_hash_fun.hash();
        return hash_value % (col * 2);
    }

    std::uint8_t countLeaderZero(const void *ptr_flow_id,
                                 std::uint64_t flow_id_len,
                                 const void *ptr_element_id,
                                 std::uint64_t element_id_len) {
        XXHash32 value_hash_fun(value_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(ptr_flow_id, flow_id_len);
        std::uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
        return hash_value;
    }

    std::uint8_t countLeaderZeroFP(uint32_t key,
                                   const void *ptr_element_id,
                                   std::uint64_t element_id_len) {
        XXHash32 value_hash_fun(value_hash_seed);
        value_hash_fun.add(ptr_element_id, element_id_len);
        value_hash_fun.add(&key, sizeof(key));
        std::uint8_t hash_value = get_leader_zero(value_hash_fun.hash());
        return hash_value;
    }

    std::string getEstimatorInfo(std::size_t selected_row) {
        std::string debugStr;
        for (std::size_t i = 0; i < col * 2; ++i) {
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

uint32_t PMatrix::FLOW_HASH_SALT = 0xdeedbeef;
uint32_t PMatrix::ELEMENT_HASH_SALT = 0x123456;
uint32_t PMatrix::VALUE_HASH_SALT = 0x765432;

#endif