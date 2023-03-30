#ifndef ATON_VHLL_AVX_HPP
#define ATON_VHLL_AVX_HPP

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
class Aton_vHLL_AVX : public TwoTupleSketch {
public:
    static const std::uint32_t HISTOGRAM_LEN = 16;
    std::uint32_t stage_num;
    std::uint32_t row;
    std::uint32_t col;
    std::uint32_t master_seed;

    int HEAP_SIZE = 256;
    int THRESHOLD = 500;

    std::vector<uint32_t> key_heap;
    std::vector<int> index_heap;
    std::vector<double> value_heap;

    std::vector<std::shared_ptr<PMatrix>> prefilters;
    std::vector<std::shared_ptr<EMatrixTC>> stages;
    Statistics op_statistics;

    Aton_vHLL_AVX(std::uint32_t stage_num, std::uint32_t row, std::uint32_t col, std::uint32_t master_seed)
        : op_statistics("../../result/op_statistics.txt") {
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

    void assign(int i, int j) {
        key_heap.at(i) = key_heap.at(j);
        index_heap.at(i) = index_heap.at(j);
        value_heap.at(i) = value_heap.at(j);
    }

    void _push_heap_aux(int first, int holeIndex, int topIndex, uint32_t key, int index, double value) {
        int parent = (holeIndex - 1) / 2;
        while (holeIndex > topIndex && value_heap.at(first + parent) > value) {
            assign(first + holeIndex, first + parent);
            holeIndex = parent;
            parent = (holeIndex - 1) / 2;
        }
        key_heap.at(first + holeIndex) = key;
        index_heap.at(first + holeIndex) = index;
        value_heap.at(first + holeIndex) = value;
    }

    void _push_heap(int first, int last) {
        _push_heap_aux(first, (last - first) - 1, 0, key_heap.at(last - 1), index_heap.at(last - 1), value_heap.at(last - 1));
    }

    void _adjust_heap(int first, int holeIndex, int len, uint32_t key, int index, double value) {
        int topIndex = holeIndex;
        int secondChild = 2 * holeIndex + 2;
        while (secondChild < len) {
            if (value_heap[first + secondChild] > value_heap[first + (secondChild - 1)])
                secondChild--;
            assign(first + holeIndex, first + secondChild);
            holeIndex = secondChild;
            secondChild = 2 * (secondChild + 1);
        }
        if (secondChild == len) {
            assign(first + holeIndex, first + (secondChild - 1));
            holeIndex = secondChild - 1;
        }
        _push_heap_aux(first, holeIndex, topIndex, key, index, value);
    }

    void _make_heap(int first, int last) {
        if (last - first < 2)
            return;
        int len = last - first;
        int parent = (len - 2) / 2;

        while (true) {
            _adjust_heap(first, parent, len, key_heap.at(first + parent), index_heap.at(first + parent), value_heap.at(first + parent));//parent下沉到合适的位置
            if (parent == 0) return;//构造完成
            parent--;
        }
    }

    void _pop_heap(int first, int last) {
        _adjust_heap(first, 0, last - 1 - first, key_heap.at(last - 1), index_heap.at(last - 1), value_heap.at(last - 1));
    }

    /* Search for the index of specific key item using AVX-512 */
    int find_index_avx512(uint32_t* filter_id, uint32_t item, int length){
        if (length == 0)
            return -1;
        const __m512i s_item = _mm512_set1_epi32(item);
        __m512i *filter = (__m512i *) filter_id;
        int batch_size = sizeof(__m512i) / sizeof(uint32_t)  ;
        int batch_num = (length - 1) / batch_size + 1 ;

        if (filter == nullptr)
            return -1;

        for (int i = 0; i != batch_num; ++i){
            __m512i a_part = _mm512_loadu_si512((__m512i*) &filter[i]);
            auto f_comp = _mm512_cmpeq_epi32_mask(s_item, a_part );
            if(f_comp) {
                int index = i * batch_size + __builtin_ctz(f_comp);
                if (index < length) {
                    return index;
                }
            }
        }
        return -1;
    }

    int find_index_avx256(uint32_t* filter_id, uint32_t item, int length){
        if (length == 0)
            return -1;
        const __m256i s_item = _mm256_set1_epi32(item);
        __m256i *filter = (__m256i *) filter_id;
        int batch_size = sizeof(__m256i) / sizeof(uint32_t)  ;
        int batch_num = (length - 1) / batch_size + 1 ;

        if (filter == nullptr)
            return -1;

        for (int i = 0; i != batch_num; ++i){
            __m256i a_part = _mm256_loadu_si256((__m256i*) &filter[i]);
            __m256i f_comp = _mm256_cmpeq_epi32(s_item, a_part );
            uint found = _mm256_movemask_epi8(f_comp);
            if(found)
                return (i * batch_size + __builtin_ctz(found) / 4);
        }
        return -1;
    }

    bool offerFlow_sketch(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {
        bool sketch_changed = false;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            std::uint32_t select_col = stage->selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
            std::uint32_t physical_col = select_col / 2;
            std::uint8_t leader_zero = stage->countLeaderZero(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
            std::uint32_t range = HISTOGRAM_LEN;
            std::uint8_t new_val;

            if (leader_zero >= range + stage->IUU.at(select_row)->base) {
                std::uint32_t delta_B = stage->IUU.at(select_row)->getMinBar();
                if (delta_B > 0) {
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

            sketch_changed = true;

            uint32_t cur_base = stage->IUU.at(select_row)->base;

            if (select_col % 2 == 0) {
                new_val <<= 4;
                stage->estimators[select_row][physical_col] &= 15; // b'00001111
                stage->estimators[select_row][physical_col] |= new_val;
                stage->IUU.at(select_row)->update(cur_val, new_val >> 4);
                stage->global_histogram.update(cur_val + cur_base, (new_val >> 4) + cur_base);
            }
            else {
                stage->estimators[select_row][physical_col] &= 240; // b'11110000
                stage->estimators[select_row][physical_col] |= new_val;
                stage->IUU.at(select_row)->update(cur_val, new_val);
                stage->global_histogram.update(cur_val + cur_base, new_val + cur_base);
            }
        }
        return sketch_changed;
    }

    /* Using sum of IUUs to estimate n_hat */
    double decodeFlow_sketch_accurate(const void *ptr_flow_id, uint64_t flow_id_len) {
        std::vector<double> res;
        std::vector<double> total_spread;
        for (auto &stage : stages) {
            double total_card = 0;
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
            for(std::uint32_t i = 0; i < stage->row; i++)
                total_card += stage->IUU.at(i)->getEstimate(stage->IUU.at(i)->base);
            total_spread.push_back(total_card);
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
    double decodeFlow_sketch_fast(const void *ptr_flow_id, uint64_t flow_id_len) {
        std::vector<double> stage_res;
        std::vector<double> stage_total_spread;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            stage_res.push_back(stage->IUU.at(select_row)->getEstimate(stage->IUU.at(select_row)->base));
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

    /* Query cardinality of a flow in the specific column from the heap */
    double decodeFlow_heap(uint32_t index) {
        std::vector<double> stage_res;
        std::vector<double> stage_total_spread;
        for (auto &prefilter : prefilters) {
            stage_res.push_back(prefilter->IUU.at(index)->getEstimate(prefilter->IUU.at(index)->base));
            stage_total_spread.push_back(prefilter->global_card.at(index));
        }
        double n_d_hat = getMedian(stage_res);
        double n_hat = getMedian(stage_total_spread);
        double factor = 1;
        if (row > 1)
            factor = (double)row / (row - 1);
        double n_f_hat = factor * (n_d_hat - n_hat / row);
        return n_f_hat;
    }

    /* Swap from pre-filter to sketch */
    void swap_out(int prefilter_index, std::vector<std::uint32_t> row_indexes) {
        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto prefilter = prefilters.at(l);
            auto stage = stages.at(l);

            std::uint32_t select_row = row_indexes.at(l);

            std::uint32_t delta_beta = prefilter->IUU.at(prefilter_index)->getMinBar();
            std::uint32_t delta_B = stage->IUU.at(select_row)->getMinBar();
            std::uint32_t newB = std::max(prefilter->IUU.at(prefilter_index)->base + delta_beta, stage->IUU.at(select_row)->base + delta_B);
            HistogramTC newQ = {col * 2,true};

            for (std::uint32_t select_col = 0; select_col < col * 2; ++select_col) {
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t src_val = prefilter->estimators[prefilter_index][physical_col];
                std::uint8_t dst_val = stage->estimators[select_row][physical_col];
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
                std::uint8_t new_val = std::min(std::max(src_sum, dst_sum) - newB, range - 1);

                if (select_col % 2 == 0) {
                    new_val <<= 4;
                    stage->estimators[select_row][physical_col] &= 15; // b'00001111
                    stage->estimators[select_row][physical_col] |= new_val;
                    newQ.histogram.at(new_val >> 4)++;
                } else {
                    stage->estimators[select_row][physical_col] &= 240; // b'11110000
                    stage->estimators[select_row][physical_col] |= new_val;
                    newQ.histogram.at(new_val)++;
                }
                stage->IUU.at(select_row)->base = newB;
                for(std::uint32_t i = 0; i < HISTOGRAM_LEN; ++i) {
                    stage->global_histogram.histogram.at(i) += newQ.histogram.at(i) - stage->IUU.at(select_row)->histogram.at(i);
                    stage->IUU.at(select_row)->histogram.at(i) = newQ.histogram.at(i);
                }
            }
            prefilter->reset_row(prefilter_index);
        }
    }

    /* Swap from sketch to pre-filter */
    void swap_in(int prefilter_index, std::vector<std::uint32_t> row_indexes) {
        for (std::uint32_t l = 0; l < stage_num; ++l) {
            auto prefilter = prefilters.at(l);
            auto stage = stages.at(l);
            prefilter->reset_row(prefilter_index);

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
            prefilter->global_card.at(prefilter_index) = stage->global_histogram.getEstimate();
        }
    }

    /* Insert <flow, element> tuple into Aton-vHLL */
    bool offerFlow(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len) {

        uint32_t key = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed);
        std::vector<std::uint32_t> row_indexes;
        for (auto &stage : stages) {
            std::uint32_t select_row = stage->selectRowIdxByFlowID(ptr_flow_id, flow_id_len);
            row_indexes.push_back(select_row);
        }

        int i = find_index_avx512(&key_heap[0],key, key_heap.size());
        // pre-filter hit
        if (i != -1)  {
            int index = index_heap.at(i);
            bool prefilter_changed = false;
            for (auto &prefilter : prefilters) {
                std::uint32_t select_col = prefilter->selectColIdxByElementID(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
                std::uint32_t physical_col = select_col / 2;
                std::uint8_t leader_zero = prefilter->countLeaderZero(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
                std::uint32_t range = HISTOGRAM_LEN;
                std::uint8_t new_val;

                // overflow detected
                if (leader_zero >= range + prefilter->IUU.at(index)->base) {
                    std::uint32_t delta_B = prefilter->IUU.at(index)->getMinBar();
                    if (delta_B > 0) {
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

                prefilter_changed = true;

                if (select_col % 2 == 0) {
                    new_val <<= 4;
                    prefilter->estimators[index][physical_col] &= 15; // b'00001111
                    prefilter->estimators[index][physical_col] |= new_val;
                    prefilter->IUU.at(index)->update(cur_val, new_val >> 4);
                } else {
                    prefilter->estimators[index][physical_col] &= 240; // b'11110000
                    prefilter->estimators[index][physical_col] |= new_val;
                    prefilter->IUU.at(index)->update(cur_val, new_val);
                }

                if(prefilter_changed) {
                    // adjust the pre-filter to restore min-heap property
                    double est = decodeFlow_heap(index);
                    value_heap.at(i) = est;
                    _make_heap(0, value_heap.size());
                }

            }
            return prefilter_changed;
        }

        bool sketch_changed = false;
        // pre-filter miss
        sketch_changed = offerFlow_sketch(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
        if (!sketch_changed) {
            return sketch_changed;
        }

        double est_card = decodeFlow_sketch_fast(ptr_flow_id, flow_id_len);

        if(est_card < THRESHOLD) {
            return sketch_changed;
        }
        if(!value_heap.empty()) {
            if(value_heap.size() == HEAP_SIZE && est_card > value_heap.at(0)) {
                int index = index_heap.at(0);
                swap_out(index, row_indexes);
                _pop_heap(0, value_heap.size());
                value_heap.pop_back();
                key_heap.pop_back();
                index_heap.pop_back();
            }
        }
        for(int j = 0; j < HEAP_SIZE; ++j) {
            if (prefilters.at(0)->global_card.at(j) == -1) {
                key_heap.emplace_back(key);
                index_heap.emplace_back(j);
                value_heap.emplace_back(est_card);

                _push_heap(0, value_heap.size());

                swap_in(j, row_indexes);
                return sketch_changed;
            }
        }
    }

    /* Query <flow, element> tuple from Aton-vHLL */
    double decodeFlow(const void *ptr_flow_id, uint64_t flow_id_len) {
        double n_f_hat = 0;
        uint32_t key = XXHash32::hash(ptr_flow_id, flow_id_len, master_seed);

        int pos = find_index_avx512(&key_heap[0],key, key_heap.size());
        // pre-filter hit
        if (pos != -1) {
            int index = index_heap.at(pos);
            n_f_hat = decodeFlow_heap(index);
            return n_f_hat;
        }

        // pre-filter miss
//        n_f_hat = decodeFlow_sketch_accurate(ptr_flow_id, flow_id_len);
        n_f_hat = decodeFlow_sketch_fast(ptr_flow_id, flow_id_len);
        return n_f_hat;
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

    void resetSketch() {
        for (auto s : stages) {
            s->reset();
        }
        key_heap.clear();
        index_heap.clear();
        value_heap.clear();
        op_statistics.reset();
    }

    void resetSeed(uint32_t new_seed) {
        this->master_seed = new_seed;
        for (std::uint32_t l = 0; l < stage_num; ++l) {
            std::uint32_t stage_seed = XXHash32::hash(&master_seed, sizeof(master_seed), l);
            stages.at(l)->resetSeed(stage_seed);
        }
    }
};

#endif