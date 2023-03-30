#ifndef ATON_VHLL_HPP
#define ATON_VHLL_HPP

#include "../utils/AccessStatics.hpp"
#include "../utils/xxhash32.h"
#include "EMatrixTC.hpp"
#include "PMatrix.hpp"
#include "TwoTupleSketch.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <memory>
#include <numeric>
#include <vector>

class Aton_vHLL : public TwoTupleSketch {
public:
    static const std::uint32_t HISTOGRAM_LEN = 16;
    std::uint32_t stage_num;
    std::uint32_t row;
    std::uint32_t col;
    std::uint32_t master_seed;
    std::uint32_t recent_insert_key;
    double recent_query_result = 0;

    int HEAP_SIZE = 256;
    int THRESHOLD = 50;

    std::vector<std::tuple<uint32_t, int, double>> min_heap;
    std::vector<std::shared_ptr<PMatrix>> prefilters;
    std::vector<std::shared_ptr<EMatrixTC>> stages;

    Statistics op_statistics;

    Aton_vHLL(std::uint32_t stage_num, std::uint32_t row, std::uint32_t col, std::uint32_t master_seed)
        : op_statistics("../../result/operation_statistics/aton_vhll_op_statistics.txt") {
        this->stage_num = stage_num;
        this->row = row;
        this->col = col;
        this->master_seed = master_seed;
        for (std::uint32_t s = 0; s < stage_num; ++s) {
            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), s);
            stages.push_back(std::make_shared<EMatrixTC>(row, col, stage_seed));
            prefilters.push_back(std::make_shared<PMatrix>(row, col, stage_seed));
        }
    }

    static bool comp (std::tuple<uint32_t, int, double> a, std::tuple<uint32_t, int, double> b) {
        return std::get<2>(a) > std::get<2>(b);
    }

    bool offerFlow_sketch(uint32_t key, const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG_SKETCH_UPDATED = false;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            std::uint32_t select_col = stage->selectColIdxByElementFP(key, ptr_element_id, element_id_len);
            std::uint32_t physical_col = select_col / 2;
            std::uint8_t leader_zero = stage->countLeaderZeroFP(key, ptr_element_id, element_id_len);
            op_statistics.addHashCnt(3);

            std::uint32_t range = HISTOGRAM_LEN;
            std::uint8_t new_val;

            // an overflow detected
            if (leader_zero >= range + stage->IUU.at(select_row)->base) {
                std::uint32_t delta_B = stage->IUU.at(select_row)->getMinBar();
                if (delta_B > 0) {
                    op_statistics.addRebaseCnt(1);
                    for (std::uint32_t j = 0; j < col * 2; ++j) {
                        if (j % 2 == 0)
                            stage->estimators[select_row][j / 2] -= delta_B * 16;
                        else
                            stage->estimators[select_row][j / 2] -= delta_B;
                    }
                    stage->IUU.at(select_row)->updateBase(delta_B);
                }
            }

            std::uint8_t cur_val = stage->estimators[select_row][physical_col];
            op_statistics.addMemoryAccess(1);

            if (select_col % 2 == 0) {
                cur_val &= 240; // b'11110000
                cur_val >>= 4;
            }
            else {
                cur_val &= 15; // b'00001111
            }

            if (leader_zero < stage->IUU.at(select_row)->base)
                continue;
            if (leader_zero - stage->IUU.at(select_row)->base >= range - 1) {
                new_val = range - 1;
            }
            else {
                new_val = leader_zero - stage->IUU.at(select_row)->base;
            }

            if (cur_val >= new_val) {
                continue;
            }

            FLAG_SKETCH_UPDATED = true;

            uint32_t cur_base = stage->IUU.at(select_row)->base;

            if (select_col % 2 == 0) {
                stage->estimators[select_row][physical_col] &= 15; // b'00001111
                stage->estimators[select_row][physical_col] |= (new_val << 4);
            }
            else {
                stage->estimators[select_row][physical_col] &= 240; // b'11110000
                stage->estimators[select_row][physical_col] |= new_val;
            }
            stage->IUU.at(select_row)->update(cur_val, new_val);
            stage->global_histogram.update(cur_val + cur_base, new_val + cur_base);

            op_statistics.addMemoryAccess(3);
        }
        if (FLAG_SKETCH_UPDATED) {
            op_statistics.addSketchUpdateCnt(1);
        }
        return FLAG_SKETCH_UPDATED;
    }

    /* Using sum of IUUs to estimate n_hat */
    double decodeFlow_sketch_accurate(uint32_t key) {
        op_statistics.addSketchQueryCnt(1);

        std::vector<double> res;
        std::vector<double> total_spread;
        for (auto &stage : stages) {
            double total_card = 0;
            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            op_statistics.addHashCnt(1);

            res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
            for(std::uint32_t i = 0; i < stage->row; i++)
                total_card += stage->IUU.at(i)->getEstimate(stage->IUU.at(i)->base);
            total_spread.push_back(total_card);
            op_statistics.addMemoryAccess(1);
        }
        double n_d_hat = getAvg(res);
        double n_hat = getAvg(total_spread);
        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_hat = factor * (n_d_hat - n_hat / row);
        return n_f_hat;
    }

    /* Using Global histogram to estimate n_hat */
    double decodeFlow_sketch_fast(uint32_t key) {
        op_statistics.addSketchQueryCnt(1);

        std::vector<double> stage_res;
        std::vector<double> stage_total_spread;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            op_statistics.addHashCnt(1);

            stage_res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
            op_statistics.addMemoryAccess(1);
            stage_total_spread.push_back(stage->global_histogram.getEstimate());
        }
        double n_d_hat = getMedian(stage_res);
        double n_hat = getMedian(stage_total_spread);
        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_hat = factor * (n_d_hat - n_hat / row);
        return n_f_hat;
    }

    double decodeFlow_heap(uint32_t index) {
        op_statistics.addHeapQueryCnt(1);

        std::vector<double> stage_res;
        std::vector<double> stage_total_spread;
        for (auto &prefilter : prefilters) {
            stage_res.push_back(prefilter->IUU.at(index)->getEstimate(prefilter->IUU.at(index)->base));
            op_statistics.addMemoryAccess(1);
            stage_total_spread.push_back(prefilter->global_card.at(index));
        }
        double n_d_hat = getAvg(stage_res);
        double n_hat = getAvg(stage_total_spread);
        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_hat = factor * (n_d_hat - n_hat / row);
        return n_f_hat;
    }

    void swap_out(int prefilter_index, uint32_t key) {
        op_statistics.addSketchUpdateCnt(1);
        op_statistics.addSwapoutCnt(1);

        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto prefilter = prefilters.at(l);
            auto stage = stages.at(l);

            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            op_statistics.addHashCnt(1);

            std::uint32_t delta_beta = prefilter->IUU.at(prefilter_index)->getMinBar();
            std::uint32_t delta_B = stage->IUU.at(select_row)->getMinBar();
            std::uint32_t newB = std::max(prefilter->IUU.at(prefilter_index)->base + delta_beta, stage->IUU.at(select_row)->base + delta_B);
            if (prefilter->IUU.at(prefilter_index)->base < newB || stage->IUU.at(select_row)->base < newB) {
                op_statistics.addRebaseCnt(1);
            }
            HistogramTC newQ(col * 2);

            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t src_val = prefilter->estimators[prefilter_index][physical_col];
                std::uint8_t dst_val = stage->estimators[select_row][physical_col];
                op_statistics.addMemoryAccess(2);

                if (select_col % 2 == 0) {
                    src_val &= 240; // b'11110000
                    src_val >>= 4;
                    dst_val &= 240;
                    dst_val >>= 4;
                } else {
                    src_val &= 15; // b'00001111
                    dst_val &= 15;
                }

                std::uint32_t range = 16;

                std::uint8_t src_sum = src_val + prefilter->IUU.at(prefilter_index)->base;
                std::uint8_t dst_sum = dst_val + stage->IUU.at(select_row)->base;

                stage->global_histogram.histogram.at(dst_sum)--;

                std::uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);

                if (select_col % 2 == 0) {
                    stage->estimators[select_row][physical_col] &= 15; // b'00001111
                    stage->estimators[select_row][physical_col] |= (new_val << 4);
                } else {
                    stage->estimators[select_row][physical_col] &= 240; // b'11110000
                    stage->estimators[select_row][physical_col] |= new_val;
                }
                op_statistics.addMemoryAccess(2);

                stage->global_histogram.histogram.at(new_val + newB)++;
                newQ.update(0, new_val);
            }
            stage->IUU.at(select_row)->base = newB;

            for(std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                stage->IUU.at(select_row)->histogram.at(i) = newQ.histogram.at(i);
            }
            op_statistics.addMemoryAccess(1);
        }
        for (auto& prefilter : prefilters) {
            prefilter->reset_row(prefilter_index);
        }
    }

    void swap_in(int prefilter_index, uint32_t key) {
        op_statistics.addHeapUpdateCnt(1);
        op_statistics.addSwapinCnt(1);

        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto prefilter = prefilters.at(l);
            auto stage = stages.at(l);
            prefilter->reset_row(prefilter_index);
            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            op_statistics.addHashCnt(1);

            prefilter->IUU.at(prefilter_index)->base = stage->IUU.at(select_row)->base;

            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t src_val = stage->estimators[select_row][physical_col];
                op_statistics.addMemoryAccess(1);

                if (select_col % 2 == 0) {
                    src_val &= 240; // b'11110000
                    src_val >>= 4;
                } else {
                    src_val &= 15; // b'00001111
                }
                std::uint8_t leader_zero = src_val + stage->IUU.at(select_row)->base;
                std::uint32_t range = 16;
                std::uint8_t new_val;

                std::uint8_t old_val = prefilter->estimators[prefilter_index][physical_col];
                op_statistics.addMemoryAccess(1);

                if (select_col % 2 == 0) {
                    old_val &= 240; // b'11110000
                    old_val >>= 4;
                } else {
                    old_val &= 15; // b'00001111
                }

                if (leader_zero < prefilter->IUU.at(prefilter_index)->base)
                    continue;
                if (leader_zero >= prefilter->IUU.at(prefilter_index)->base + range - 1) {
                    new_val = range - 1;
                } else {
                    new_val = leader_zero - prefilter->IUU.at(prefilter_index)->base;
                }

                if (old_val >= new_val) {
                    continue;
                }

                if (select_col % 2 == 0) {
                    new_val <<= 4;
                    prefilter->estimators[prefilter_index][physical_col] &= 15; // b'00001111
                    prefilter->estimators[prefilter_index][physical_col] |= new_val;
                    prefilter->IUU.at(prefilter_index)->update(old_val, new_val >> 4);
                } else {
                    prefilter->estimators[prefilter_index][physical_col] &= 240; // b'11110000
                    prefilter->estimators[prefilter_index][physical_col] |= new_val;
                    prefilter->IUU.at(prefilter_index)->update(old_val, new_val);
                }
                op_statistics.addMemoryAccess(3);
            }
            prefilter->global_card.at(prefilter_index) = stage->global_histogram.getEstimate();
        }
    }

    void swap_in_merge(Aton_vHLL src_sketch, int prefilter_index, uint32_t key) {
        op_statistics.addSwapinCnt(1);
        std::vector<std::uint32_t> row_indexes;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowFP(key);
            row_indexes.push_back(select_row);
        }

        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto prefilter = src_sketch.prefilters.at(l);
            auto stage = stages.at(l);
            //            prefilter->reset_row(prefilter_index);

            std::uint32_t select_row = row_indexes.at(l);
            prefilter->IUU.at(prefilter_index)->base = stage->IUU.at(select_row)->base;

            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t src_val = stage->estimators[select_row][physical_col];
                if (select_col % 2 == 0) {
                    src_val &= 240; // b'11110000
                    src_val >>= 4;
                } else {
                    src_val &= 15; // b'00001111
                }
                std::uint8_t leader_zero = src_val + stage->IUU.at(select_row)->base;
                std::uint32_t range = 16;
                std::uint8_t new_val;

                std::uint8_t old_val = prefilter->estimators[prefilter_index][physical_col];
                if (select_col % 2 == 0) {
                    old_val &= 240; // b'11110000
                    old_val >>= 4;
                } else {
                    old_val &= 15; // b'00001111
                }

                if (leader_zero < prefilter->IUU.at(prefilter_index)->base)
                    continue;
                if (leader_zero >= prefilter->IUU.at(prefilter_index)->base + range - 1) {
                    new_val = range - 1;
                } else {
                    new_val = leader_zero - prefilter->IUU.at(prefilter_index)->base;
                }

                if (old_val >= new_val) {
                    continue;
                }

                if (select_col % 2 == 0) {
                    new_val <<= 4;
                    prefilter->estimators[prefilter_index][physical_col] &= 15; // b'00001111
                    prefilter->estimators[prefilter_index][physical_col] |= new_val;
                    prefilter->IUU.at(prefilter_index)->update(old_val, new_val >> 4);
                } else {
                    prefilter->estimators[prefilter_index][physical_col] &= 240; // b'11110000
                    prefilter->estimators[prefilter_index][physical_col] |= new_val;
                    prefilter->IUU.at(prefilter_index)->update(old_val, new_val);
                }
            }
            prefilter->global_card.at(prefilter_index) += stage->global_histogram.getEstimate();
        }
    }


    void mergePrefilter(Aton_vHLL &src_sketch, int src_index, int dst_index) {
        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto src_stage = src_sketch.prefilters.at(l);
            auto dst_stage = prefilters.at(l);


            uint32_t src_select_row = src_index;
            uint32_t dst_select_row = dst_index;

            std::uint32_t delta_B1 = src_stage->IUU.at(src_select_row)->getMinBar();
            std::uint32_t delta_B2 = dst_stage->IUU.at(dst_select_row)->getMinBar();
            std::uint32_t newB = std::max(src_stage->IUU.at(src_select_row)->base + delta_B1, dst_stage->IUU.at(dst_select_row)->base + delta_B2);
            HistogramTC newQ = {col * 2};

            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t src_val = src_stage->estimators[src_select_row][physical_col];
                std::uint8_t dst_val = dst_stage->estimators[dst_select_row][physical_col];
                if (select_col % 2 == 0) {
                    src_val &= 240; // b'11110000
                    src_val >>= 4;
                    dst_val &= 240;
                    dst_val >>= 4;
                } else {
                    src_val &= 15; // b'00001111
                    dst_val &= 15;
                }

                std::uint32_t range = 16;

                std::uint8_t src_sum = src_val + src_stage->IUU.at(src_select_row)->base;
                std::uint8_t dst_sum = dst_val + dst_stage->IUU.at(dst_select_row)->base;
                std::uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);

                if (select_col % 2 == 0) {
                    new_val <<= 4;
                    dst_stage->estimators[dst_select_row][physical_col] &= 15; // b'00001111
                    dst_stage->estimators[dst_select_row][physical_col] |= new_val;
                    newQ.update(0, new_val >> 4);
                } else {
                    dst_stage->estimators[dst_select_row][physical_col] &= 240; // b'11110000
                    dst_stage->estimators[dst_select_row][physical_col] |= new_val;
                    newQ.update(0, new_val);
                }
                dst_stage->IUU.at(dst_select_row)->base = newB;
                for(std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                    dst_stage->IUU.at(dst_select_row)->histogram.at(i) = newQ.histogram.at(i);
                }

            }
            dst_stage->global_card.at(dst_select_row) += src_stage->global_card.at(src_select_row);

        }
    }

    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        uint32_t key = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed);
        op_statistics.addHashCnt(1);

        for(auto & i : min_heap) {
            // pre-filter hit
            if (std::get<0>(i) == key) {
                int index = std::get<1>(i);
                bool FLAG_PREFILTER_UPDATED = false;
                for (auto& prefilter : prefilters) {
                    std::uint32_t select_col = prefilter->selectColIdxByElementFP(key, ptr_element_id, element_id_len);
                    std::uint32_t physical_col = select_col / 2;
                    std::uint8_t leader_zero = prefilter->countLeaderZeroFP(key, ptr_element_id, element_id_len);
                    std::uint32_t range = HISTOGRAM_LEN;
                    std::uint8_t new_val;
                    op_statistics.addHashCnt(2);

                    // an overflow detected
                    if (leader_zero >= range + prefilter->IUU.at(index)->base) {
                        std::uint32_t delta_B = prefilter->IUU.at(index)->getMinBar();
                        if (delta_B > 0) {
                            op_statistics.addRebaseCnt(1);
                            for (std::uint32_t j = 0; j < col * 2; ++j) {
                                if (j % 2 == 0)
                                    prefilter->estimators[index][j / 2] -= delta_B * 16;
                                else
                                    prefilter->estimators[index][j / 2] -= delta_B;
                            }
                            prefilter->IUU.at(index)->updateBase(delta_B);
                        }
                    }

                    std::uint8_t cur_val = prefilter->estimators[index][physical_col];
                    op_statistics.addMemoryAccess(1);

                    if (select_col % 2 == 0) {
                        cur_val &= 240; // b'11110000
                        cur_val >>= 4;
                    } else {
                        cur_val &= 15; // b'00001111
                    }

                    if (leader_zero < prefilter->IUU.at(index)->base)
                        continue;
                    if (leader_zero - prefilter->IUU.at(index)->base >= range - 1) {
                        new_val = range - 1;
                    } else {
                        new_val = leader_zero - prefilter->IUU.at(index)->base;
                    }

                    if (cur_val >= new_val) {
                        continue;
                    }

                    FLAG_PREFILTER_UPDATED = true;

                    if (select_col % 2 == 0) {
                        prefilter->estimators[index][physical_col] &= 15; // b'00001111
                        prefilter->estimators[index][physical_col] |= (new_val << 4);
                    } else {
                        prefilter->estimators[index][physical_col] &= 240; // b'11110000
                        prefilter->estimators[index][physical_col] |= new_val;
                    }
                    prefilter->IUU.at(index)->update(cur_val, new_val);

                    op_statistics.addMemoryAccess(3);

                }

                if(FLAG_PREFILTER_UPDATED) {
                    op_statistics.addHeapUpdateCnt(1);

                    // adjust the pre-filter to restore min-heap property
                    double est = decodeFlow_heap(index);
                    std::get<2>(i) = est;
                    std::make_heap(min_heap.begin(), min_heap.end(), comp);

                    recent_insert_key = key;
                    recent_query_result = est;
                }

                return FLAG_PREFILTER_UPDATED;
            }
        }


        // pre-filter miss
        bool FLAG_SKETCH_UPDATED = false;
        FLAG_SKETCH_UPDATED = offerFlow_sketch(key, ptr_element_id, element_id_len);
        if (!FLAG_SKETCH_UPDATED) {
            return FLAG_SKETCH_UPDATED;
        }

        double est_card = decodeFlow_sketch_fast(key);
        recent_insert_key = key;
        recent_query_result = est_card;
        if(est_card < THRESHOLD) {
            return FLAG_SKETCH_UPDATED;
        }
        if(!min_heap.empty()) {
            double top_element_value = std::get<2>(min_heap.at(0));
            if(min_heap.size() == HEAP_SIZE && est_card > top_element_value) {
                uint32_t top_element_key = std::get<0>(min_heap.at(0));
                int top_element_index = std::get<1>(min_heap.at(0));
                swap_out(top_element_index, top_element_key);
                std::pop_heap(min_heap.begin(),min_heap.end(), comp);
                min_heap.pop_back();
            }
        }
        for(int pos = 0; pos < HEAP_SIZE; ++pos) {
            if (prefilters.at(0)->global_card.at(pos) == -1) {
                auto temp = std::make_tuple(key, pos, est_card);
                min_heap.emplace_back(temp);
                std::push_heap(min_heap.begin(),min_heap.end(), comp);
                swap_in(pos, key);
                return FLAG_SKETCH_UPDATED;
            }
        }
        return FLAG_SKETCH_UPDATED;
    }


    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
        uint32_t key = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed);
        op_statistics.addHashCnt(1);

        if (key == recent_insert_key) {
            return recent_query_result;
        }

        double n_f_hat = 0;

        for(auto & i : min_heap) {
            // pre-filter hit
            if (std::get<0>(i) == key) {
                int index = std::get<1>(i);
                n_f_hat = decodeFlow_heap(index);
                return n_f_hat;
            }
        }

        // pre-filter miss
        n_f_hat = decodeFlow_sketch_accurate(key);
//        n_f_hat = decodeFlow_sketch_fast(key);
        return n_f_hat;
    }

    void mergeFlow(Aton_vHLL src_sketch) {
        for(auto & b : src_sketch.min_heap) {
            bool hit = false;
            for(auto & a : min_heap) {
                if (std::get<0>(a) == std::get<0>(b)) {
                    hit = true;

                    mergePrefilter(src_sketch, std::get<1>(b), std::get<1>(a));
                    double est = decodeFlow_heap(std::get<1>(a));
                    std::get<2>(a) = est;
                    std::make_heap(min_heap.begin(), min_heap.end(), comp);

                    break;
                }
            }

            if (!hit) {
                double est_card = src_sketch.decodeFlow_heap(std::get<1>(b));
                /* 0722新增 */
                //                est_card += decodeFlow_sketch(std::get<0>(b));
                //                swap_in_merge(src_sketch, std::get<1>(b), std::get<0>(b));
                /* 0722新增 */
                if(!min_heap.empty()) {
                    // 大于min_heap a的最小值，替换
                    if(min_heap.size() == HEAP_SIZE && est_card > std::get<2>(min_heap.at(0))) {
                        int index = std::get<1>(min_heap.at(0));
                        swap_out(index, std::get<0>(min_heap.at(0)));
                        std::pop_heap(min_heap.begin(),min_heap.end(), comp);
                        min_heap.pop_back();
                    }
                    else if (min_heap.size() == HEAP_SIZE && est_card < std::get<2>(min_heap.at(0))) {
                        src_sketch.swap_out(std::get<1>(b), std::get<0>(b));
                        continue;
                    }
                }
                for(int j = 0; j < HEAP_SIZE; ++j) {
                    if (prefilters.at(0)->global_card.at(j) == -1) {
                        auto src_index = std::get<1>(b);
                        auto temp = std::make_tuple(std::get<0>(b), j, est_card);
                        min_heap.emplace_back(temp);
                        std::push_heap(min_heap.begin(),min_heap.end(), comp);

                        for (std::uint32_t l = 0; l < stage_num; ++l) {
                            auto dst_prefilter = prefilters.at(l);
                            auto src_prefilter = src_sketch.prefilters.at(l);
                            dst_prefilter->reset_row(j);

                            std::uint32_t select_row = src_index;
                            dst_prefilter->IUU.at(j)->base = src_prefilter->IUU.at(select_row)->base;

                            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                                std::uint32_t physical_col = select_col / 2;
                                std::uint8_t src_val = src_prefilter->estimators[select_row][physical_col];
                                if (select_col % 2 == 0) {
                                    src_val &= 240; // b'11110000
                                    src_val >>= 4;
                                } else {
                                    src_val &= 15; // b'00001111
                                }
                                std::uint8_t leader_zero = src_val + src_prefilter->IUU.at(select_row)->base;
                                std::uint32_t range = 16;
                                std::uint8_t new_val;

                                std::uint8_t old_val = 0;

                                if (leader_zero < dst_prefilter->IUU.at(j)->base)
                                    continue;
                                if (leader_zero >= dst_prefilter->IUU.at(j)->base + range - 1) {
                                    new_val = range - 1;
                                } else {
                                    new_val = leader_zero - dst_prefilter->IUU.at(j)->base;
                                }

                                if (old_val >= new_val) {
                                    continue;
                                }

                                if (select_col % 2 == 0) {
                                    new_val <<= 4;
                                    dst_prefilter->estimators[j][physical_col] &= 15; // b'00001111
                                    dst_prefilter->estimators[j][physical_col] |= new_val;
                                    dst_prefilter->IUU.at(j)->update(old_val, new_val >> 4);
                                } else {
                                    dst_prefilter->estimators[j][physical_col] &= 240; // b'11110000
                                    dst_prefilter->estimators[j][physical_col] |= new_val;
                                    dst_prefilter->IUU.at(j)->update(old_val, new_val);
                                }
                            }
                            dst_prefilter->global_card.at(j) = src_prefilter->global_card.at(src_index);
                        }

                        break;
                    }
                }
            }

        }

        //        for(auto & a : min_heap) {
        //            bool hit = false;
        //            for(auto & b : src_sketch.min_heap) {
        //                if (std::get<0>(a) == std::get<0>(b)) {
        //                    hit = true;
        //
        //                    break;
        //                }
        //            }
        //            if (!hit) {
        //                swap_in_merge(*this, std::get<1>(a), std::get<0>(a));
        //            }
        //        }

        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto src_stage = src_sketch.stages.at(l);
            auto dst_stage = stages.at(l);
            for (std::uint32_t select_row = 0; select_row < row; ++select_row) {
                std::uint32_t delta_B_src = src_stage->IUU.at(select_row)->getMinBar();
                std::uint32_t delta_B_dst = dst_stage->IUU.at(select_row)->getMinBar();
                std::uint32_t newB = std::max(src_stage->IUU.at(select_row)->base + delta_B_src, dst_stage->IUU.at(select_row)->base + delta_B_dst);
                HistogramTC newQ = {col * 2};

                for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                    std::uint32_t physical_col = select_col / 2;
                    std::uint8_t src_val = src_stage->estimators[select_row][physical_col];
                    std::uint8_t dst_val = dst_stage->estimators[select_row][physical_col];
                    if (select_col % 2 == 0) {
                        src_val &= 240; // b'11110000
                        src_val >>= 4;
                        dst_val &= 240;
                        dst_val >>= 4;
                    } else {
                        src_val &= 15; // b'00001111
                        dst_val &= 15;
                    }

                    std::uint32_t range = 16;


                    std::uint8_t src_sum = src_val + src_stage->IUU.at(select_row)->base;
                    std::uint8_t dst_sum = dst_val + dst_stage->IUU.at(select_row)->base;

                    dst_stage->global_histogram.histogram.at(src_sum)--;

                    std::uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);


                    if (select_col % 2 == 0) {
                        dst_stage->estimators[select_row][physical_col] &= 15; // b'00001111
                        dst_stage->estimators[select_row][physical_col] |= (new_val << 4);
                    }
                    else {
                        dst_stage->estimators[select_row][physical_col] &= 240; // b'11110000
                        dst_stage->estimators[select_row][physical_col] |= new_val;
                    }

                    dst_stage->global_histogram.histogram.at(new_val + newB)++;
                    newQ.update(0, new_val);
                    dst_stage->IUU.at(select_row)->base = newB;

                    for(std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                        dst_stage->IUU.at(select_row)->histogram.at(i) = newQ.histogram.at(i);
                    }
                }
            }
        }
    }

    void flowTrace(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        bool FLAG = false;
        FLAG = offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        if (FLAG) {
            decodeFlow(ptr_flow_id, flow_id_len);
        }
    }

    inline double getMedian(std::vector<double> &v) {
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

    inline double getAvg(std::vector<double> &v) {
        std::uint32_t len = v.size();
        if (len == 0) {
            return 0;
        }
        double sum = accumulate(begin(v), end(v), 0.0);
        double avg = sum / (double)v.size();

        return avg;
    }

    void saveStatistic() { op_statistics.save_to_file(); }

    void resetSketch() {
        for (auto s : stages) {
            s->reset();
        }
        for (auto p : prefilters) {
            p->reset();
        }
        min_heap.clear();
        op_statistics.reset();
    }

    void resetSeed(uint32_t new_seed) {
        this->master_seed = new_seed;
        for (std::uint32_t l = 0; l < stage_num; ++l) {
            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), l);
            stages.at(l)->resetSeed(stage_seed);
            prefilters.at(l)->resetSeed(stage_seed);
        }
    }
};

#endif