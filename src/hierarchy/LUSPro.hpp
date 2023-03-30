//
// Created by Masshiro on 2022/9/21.
//

#ifndef CARDINALITYMOMENTESTIMATOR_LUSPRO_HPP
#define CARDINALITYMOMENTESTIMATOR_LUSPRO_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <type_traits>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <climits>

#include "../sketch/On_vHLL.hpp"
//#include "../sketch/Ton_vHLL.hpp"
#include "../utils/HierarchyUtils.hpp"
#include "../filter/MinheapFilter.hpp"
#include "fmt/core.h"

namespace hierarchy{
    template<class SKETCH>
    class LUSSketch{
        SKETCH ** sketch;
        hierarchy::MinHeapFilter<std::string> filter;
        std::unordered_map<std::string, std::unordered_map<std::string, bool> > hash_table;

        int * count_level_pkts;

        uint32_t m_seed;
        uint32_t m_level_count;     // level count of sketches, hash table is NOT included.
        uint32_t m_filter_size;
        uint32_t m_sketch_rows;
        double m_progressive_rate;
        std::vector<double> moment_for_each_level;

    public:
        LUSSketch(uint32_t top_k, uint32_t level_num, uint32_t sketch_stages,
                  uint32_t sketch_rows, uint32_t sketch_cols, uint32_t sketch_seed, double prog_rate = 2) : filter(top_k, level_num){
            m_filter_size = top_k;
            m_level_count = level_num;
            m_seed = sketch_seed;
            m_sketch_rows = sketch_rows;
            m_progressive_rate = prog_rate;

            count_level_pkts = new int [m_level_count + 1];
            for (int i = 0; i < m_level_count + 1; ++i) {
                count_level_pkts[i] = 0;
            }
            sketch = new SKETCH * [m_level_count];

            if (level_num > 0){
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_int_distribution<uint32_t> dist(1, 1e6);
                for (uint32_t j = 0; j != m_level_count; ++j){
                    uint32_t seed = dist(mt);
                    sketch[j] = new SKETCH(sketch_stages, sketch_rows/ pow(prog_rate,j), sketch_cols, seed);
//                    sketch[j] = new SKETCH(sketch_stages, sketch_rows, sketch_cols, seed);
                }
            }
        }

        int memory_usage_int_bits() {
            int bytes_used = 0;
            int bits_used = 0;
            // sketch and filter
            for (int i = 0; i < m_level_count; ++i) {
                bits_used += sketch[i]->memory_usage_in_bits();
                bytes_used += (sizeof(int) + sizeof(double)) * m_filter_size;
            }
            // hash table
            bytes_used += sizeof(int) * 2 * hash_table.size();

            return bytes_used * CHAR_BIT + bits_used;
        }

        std::vector<std::pair<double, double>> diplay_filter_cards_ranges() {
            std::vector<std::pair<double, double>> res = filter.get_card_range_for_levels();
            for (auto p: res) {
                std::cout << p.first << ' ' << p.second << std::endl;
            }
        }

        int get_hash_table_size() {
            return hash_table.size();
        }

        ~LUSSketch() {
            for (uint32_t i = 0; i < m_level_count; ++i) {
                delete sketch[i];
            }
            delete sketch;
            delete count_level_pkts;
        }

        static inline uint32_t get_leading_zeros(uint32_t hash_value) {
            uint32_t mask = 1;
            uint32_t cnt = 1;
            // uint32_t(int)为4字节长度，sizeof(hash_value) * 8 共32bit
            for ( uint32_t i = 0; i < sizeof(hash_value) * 8; i++) {
                // 找到hash value二进制串最右的1的位置
                if ((mask & hash_value) != 0){
                    break;
                }
                mask <<= 1;
                cnt++;
            }
            if (cnt >= 32) {
                cnt = 31;
            }
            // 返回值cnt的范围是1-31，可以在5bit内表示
            return cnt;
        }

        int get_sample_level(uint32_t hash_val) {
//            return get_leading_ones(hash_val, m_level_count - 1) + 1;
//            int lead_cnt = get_leading_ones(hash_val, 31);
            int lead_cnt = get_leading_zeros(hash_val);
            if (lead_cnt % (int(m_progressive_rate) / 2) != 0) return -1;
            int idx = lead_cnt / (m_progressive_rate / 2);
            return std::min(idx, int(m_level_count));
        }

        void display_moments_for_each_level() {
            for (auto m: moment_for_each_level) {
                std::cout << m << ' ';
            }
            std::cout << std::endl;
        }

        void display_filter_at_level_k(int k) {
            filter.print_list_at_level_k(k);
        }

        void update_hash_table(std::string, std::string);

        void update_sketch_at_level_k(std::string, std::string, uint32_t);

        bool update_filter_at_level_k(std::string, double, uint32_t);

        void update(std::string, std::string);

        void display_pkts_each_level();

        double calculate_moment_power(double(*f)(double x, uint8_t p), uint8_t);

        double calculate_moment_entropy(double(*f)(double x));
    };

//void LUSSketch::display_sketch_prefilter_sizes() {
//    for (int i = 0; i < m_level_count; ++i) {
//        std::cout << "\nLevel " << i << std::endl;
//        std::cout << sketch[i]->getPreFilterFlowNum();
//    }
//}

    template<class SKETCH>
    void LUSSketch<SKETCH>::update_hash_table(std::string flow_id, std::string element_id) {
        count_level_pkts[m_level_count]++;
        auto position = hash_table.find(flow_id);
        if (position == hash_table.end()) {     //  this flow id has not been inserted
            hash_table[flow_id].insert(std::make_pair(element_id, true));
        } else {                                    //  this flow id has been inserted
            if (position->second.find(element_id) == position->second.end()) {
                //  the element is unique, we need to insert it
                position->second.insert(std::make_pair(element_id, true));
            }
        }
    }

    template<class SKETCH>
    void LUSSketch<SKETCH>::update_sketch_at_level_k(std::string flow_id, std::string element_id, uint32_t k) {
        count_level_pkts[k]++;
        sketch[k]->offerFlow(flow_id.c_str(), flow_id.length(), element_id.c_str(), element_id.length());
    }

    template<class SKETCH>
    bool LUSSketch<SKETCH>::update_filter_at_level_k(std::string ID, double card, uint32_t k) {
        return filter.insert_element_at_level_k(ID, card, k);
    }

    template<class SKETCH>
    void LUSSketch<SKETCH>::update(std::string flow_id, std::string element_id) {
        if (m_level_count == 0){
            update_hash_table(flow_id, element_id);
            return;
        }

        XXHash32 hash32(m_seed);
        hash32.add(flow_id.c_str(), flow_id.length());
        uint32_t flow_id_hash = hash32.hash();
        uint32_t j_star = get_sample_level(flow_id_hash);
        //floor(get_leading_ones(flow_id_hash, m_level_count) / (m_progressive_rate / 2.0));  //  it is a INDEX!!
        bool is_HH;

        //  Now matter what, update the sketch at level 0, so that the L1-moment will be unbiased.
        update_sketch_at_level_k(flow_id, element_id, 0);
        update_filter_at_level_k(flow_id, sketch[0]->decodeFlow(flow_id.c_str(), flow_id.length()), 0);

        if (j_star == -1) return;
//        update_filter_at_level_k(flow_id, sketch[0]->decodeFlow(flow_id.c_str(), flow_id.length()), 0);
//        if (j_star != 0) {  //  in case the packet is repeatedly recorded.
//            update_sketch_at_level_k(flow_id, element_id, 0);
//            count_level_pkts[0]--;
//        }

        if (j_star == m_level_count){
            update_hash_table(flow_id, element_id);
            for (int k = j_star-1; k != -1; --k){
                filter.insert_element_at_level_k(flow_id, hash_table[flow_id].size(), k);
            }
        }
        else if(j_star != 0) {
            update_sketch_at_level_k(flow_id, element_id, j_star);
            double estimate_value = sketch[j_star]->decodeFlow(flow_id.c_str(), flow_id.length());

            if (estimate_value > 0.0) {
                is_HH = update_filter_at_level_k(flow_id, estimate_value, j_star);
                if (is_HH) {
                    for (int k = j_star-1; k >= 0; --k) {                                                                  //k != -1
                        update_filter_at_level_k(flow_id, estimate_value, k);
                    }
                }
            }
        }
    }

    template<class SKETCH>
    void LUSSketch<SKETCH>::display_pkts_each_level() {
        int base = count_level_pkts[0];
        for (int i = 0; i < m_level_count + 1; ++i) {
            std::cout << "\t -> level " << i << ": " << count_level_pkts[i] << "\tpercentage: "
            << double(count_level_pkts[i]) /double(base) << "(" << 1/ pow(m_progressive_rate,i) << ")"
            << std::endl;
        }
    }

    template<class SKETCH>
    double LUSSketch<SKETCH>::calculate_moment_power(double (*f)(double, uint8_t), uint8_t power) {
        if (power == 1 && m_level_count != 0) {
            return sketch[0]->getNhatSlow();
        }

        double below_level_moment = 0.0;
        std::pair<std::string, double>* ptr_level_k;
//        uint32_t w = m_sketch_rows;

        for (auto flow: hash_table) {
            below_level_moment += f(flow.second.size(), power);
        }

        if (m_level_count == 0) {
            return below_level_moment;
        }

        for (int k = m_level_count - 1; k > 0; --k) {
            below_level_moment *= m_progressive_rate;
            ptr_level_k = filter.get_elements_at_level_k(k);
            double n_fast = sketch[k]->getNhatFast();
            double n_slow = sketch[k]->getNhatSlow();
            uint32_t w = m_sketch_rows / pow(m_progressive_rate, k);

            for (int j = 0; j < filter.get_storage_condition_at_level_k(k); ++j) {
                std::string flow_id = ptr_level_k[j].first;
                XXHash32 hash32(m_seed);
                hash32.add(flow_id.c_str(), flow_id.length());
                uint32_t rnd = hash32.hash();
                uint32_t deepest_level = get_sample_level(rnd);

                int hf = 0;

                if (k < deepest_level) {
                    hf = 1;
                }

                if (ptr_level_k[j].second != 0.0) {
                    double corrected_val = ptr_level_k[j].second + (1/(w-1)) * (n_fast - n_slow);
                    below_level_moment += (f(corrected_val, power) * (1-m_progressive_rate*hf));
//                below_level_moment += (f(ptr_level_k[j].second, power) * (1-2*hf));
                }
            }
//            moment_for_each_level.emplace_back(below_level_moment);
        }

        below_level_moment *= 2;
        ptr_level_k = filter.get_elements_at_level_k(0);
        double n_fast = sketch[0]->getNhatFast();
        double n_slow = sketch[0]->getNhatSlow();
        uint32_t w = m_sketch_rows / pow(m_progressive_rate, 0);
        for (int j = 0; j < filter.get_storage_condition_at_level_k(0); ++j) {
            std::string flow_id = ptr_level_k[j].first;
            XXHash32 hash32(m_seed);
            hash32.add(flow_id.c_str(), flow_id.length());
            uint32_t rnd = hash32.hash();
            uint32_t deepest_level = get_sample_level(rnd);

           if (deepest_level == -1) {
               double corrected_val = ptr_level_k[j].second + (1/(w-1)) * (n_fast - n_slow);
               below_level_moment += f(corrected_val, power);
           }
           else if (deepest_level > 0) {
               double corrected_val = ptr_level_k[j].second + (1/(w-1)) * (n_fast - n_slow);
               below_level_moment -= f(corrected_val, power);
           }
        }


        return below_level_moment;
    }

//    template<class SKETCH>
//    double LUSSketch<SKETCH>::calculate_moment_entropy(double (*f)(double)) {
//        double below_level_moment = 0.0;
//        std::pair<std::string, double>* ptr_level_k;
////        uint32_t w = m_sketch_rows;
//
//        for (auto flow: hash_table) {
//            below_level_moment += f(flow.second.size());
//        }
//
//        if (m_level_count == 0) {
//            return below_level_moment;
//        }
//
//        for (int k = m_level_count - 1; k != -1; --k) {
//            below_level_moment *= 2;
//            ptr_level_k = filter.get_elements_at_level_k(k);
//            double n_fast = sketch[k]->getNhatFast();
//            double n_slow = sketch[k]->getNhatSlow();
//            uint32_t w = m_sketch_rows / pow(2, k);
//
//            for (int j = 0; j < filter.get_storage_condition_at_level_k(k); ++j) {
//                std::string flow_id = ptr_level_k[j].first;
//                XXHash32 hash32(m_seed);
//                hash32.add(flow_id.c_str(), flow_id.length());
//                uint32_t rnd = hash32.hash();
//                uint32_t deepest_level = get_leading_ones(rnd, m_level_count);
//
//                int hf = 0;
//
//                if (k < deepest_level) {
//                    hf = 1;
//                }
//
//                if (ptr_level_k[j].second != 0.0) {
//                    double corrected_val = ptr_level_k[j].second + (1/(w-1)) * (n_fast - n_slow);
//                    below_level_moment += (f(corrected_val) * (1-2*hf));
//                }
//            }
//            moment_for_each_level.emplace_back(below_level_moment);
//        }
//        return below_level_moment;
//    }

}

#endif //CARDINALITYMOMENTESTIMATOR_LUSPRO_HPP
