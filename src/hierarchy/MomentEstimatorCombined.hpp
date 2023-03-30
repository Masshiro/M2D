//
// Created by Masshiro on 2023/3/23.
//

#ifndef CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORCOMBINED_HPP
#define CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORCOMBINED_HPP

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
#include <bitset>
#include <chrono>

#include "../sketch/On_vHLL.hpp"
#include "../sketch/Ton_vHLL.hpp"
#include "../utils/HierarchyUtils.hpp"
#include "../filter/MinheapFilter.hpp"
#include "../filter/MapImplFilter.hpp"
#include "fmt/core.h"

using std::string, std::vector, std::map, std::bitset;
using std::unordered_map;
using std::cout, std::endl;
using namespace std::chrono;

namespace hierarchy {
    class MomentEstimatorCombined{
    private:
        On_vHLL ** sketches;
        hierarchy::MapImplFilter<string> filters;
        int m_seed;
        int m_level_count;
        int m_filter_size;
        double m_prog_rate;
        string sys_type; // UnivMon, LUS, M2D

    public:
        MomentEstimatorCombined(int top_k, int level_num, int sketch_stages, int sketch_rows, int sketch_cols,
                                int master_seed, double prog_rate, string name) : filters(top_k, level_num) {
            m_filter_size = top_k;
            m_level_count = level_num;
            m_seed = master_seed;
            sys_type = name;

            if (sys_type == "M2D") {
                m_prog_rate = prog_rate;
            } else {
                m_prog_rate = 2;
            }

            sketches = new On_vHLL* [m_level_count];
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_int_distribution<uint32_t> dist(1, 1e6);
            for (uint32_t j = 0; j != m_level_count; ++j){
                uint32_t seed = dist(mt);
                if (sys_type == "M2D") {
                    sketches[j] = new On_vHLL(sketch_stages, sketch_rows / pow(prog_rate,j), sketch_cols, seed);
                } else {
                    sketches[j] = new On_vHLL(sketch_stages, sketch_rows, sketch_cols, seed);
                }
            }
        }

        ~MomentEstimatorCombined() {
            for (uint32_t i = 0; i < m_level_count; ++i) {
                delete sketches[i];
            }
            delete sketches;
        }

        // for UnivMon
        int get_sample_level1(uint32_t hash_val) {
            uint32_t mask = 1;
            for (int i = 0; i != m_level_count-1; ++i){
                if ((hash_val & mask) != 0){
                    mask = mask<<1;
                } else {
                    return i;
                }
            }
            return m_level_count-1;
        }

        // for LUS & M2D
        int get_sample_level2(uint32_t hash_val) {
            // return value: [1, l-1]
            bitset<32> bi_hash_val = hash_val;
            int lead_cnt = 0;
            for (; lead_cnt < 32; ++lead_cnt) {
                if (bi_hash_val[lead_cnt] == 1) break;
            }
            ++lead_cnt;
            if (lead_cnt % (int(m_prog_rate) / 2) != 0) return -1;
            int idx = lead_cnt / (m_prog_rate / 2);
            return std::min(idx, int(m_level_count-1));
        }

        void update_sketch_at_level_k(string, string, int);

        bool update_filter_at_level_k(string, double, int);

        double update(vector<pair<string, string>>&);

        void update1(vector<pair<string, string>>&);

        void update2(vector<pair<string, string>>&);

    };

    bool MomentEstimatorCombined::update_filter_at_level_k(string ID, double card, int k) {
        return filters.insert_element_at_level_k(ID, card, k);
    }

    void MomentEstimatorCombined::update_sketch_at_level_k(string flow_id, string element_id, int k) {
        sketches[k]->offerFlow(flow_id.c_str(), flow_id.length(), element_id.c_str(), element_id.length());
    }

    void MomentEstimatorCombined::update1(vector<pair<string, string>> & tuple_pairs) {
//        int cnt = 0;
        for (auto one_tuple: tuple_pairs) {
            XXHash32 hash32(m_seed);
            hash32.add(one_tuple.first.c_str(), one_tuple.first.length());
            int j_star = get_sample_level1(hash32.hash());

//            cout << "updating " << one_tuple.first << ' ' << one_tuple.second << " (" << cnt++ << " / " << tuple_pairs.size() << ")" << endl;

            for (int k = j_star; k >= 0; --k) {
                update_sketch_at_level_k(one_tuple.first, one_tuple.second, k);
                double esti_val = sketches[k]->decodeFlow(one_tuple.first.c_str(), one_tuple.first.length());
                if (esti_val > 0) {
                    update_filter_at_level_k(one_tuple.first, esti_val, k);
                } else {
                    continue;
                }
            }
        }
    }

    void MomentEstimatorCombined::update2(vector<pair<string, string>> & tuple_pairs) {
        for (auto one_tuple: tuple_pairs) {
            XXHash32 hash32(m_seed);
            hash32.add(one_tuple.first.c_str(), one_tuple.first.length());
            int j_star = get_sample_level2(hash32.hash());

            update_sketch_at_level_k(one_tuple.first, one_tuple.second, 0);
            update_filter_at_level_k(one_tuple.first, sketches[0]->decodeFlow(one_tuple.first.c_str(), one_tuple.first.length()), 0);

            bool is_hh;

            if (j_star == -1) continue;
            else {
                update_sketch_at_level_k(one_tuple.first, one_tuple.second, j_star);
                double esti_val = sketches[j_star]->decodeFlow(one_tuple.first.c_str(), one_tuple.first.length());
                if (esti_val > 0) {
                    is_hh = update_filter_at_level_k(one_tuple.first, esti_val, j_star);
                    if (is_hh) {
                        for (int k = j_star-1; k >= 0; --k) {
                            update_filter_at_level_k(one_tuple.first, esti_val, k);
                        }
                    }
                }
            }
        }
    }


    double MomentEstimatorCombined::update(vector<pair<string, string>>& tuple_pairs) {
        auto start = system_clock::now();

        if (sys_type == "UnivMon") {
            cout << "updateing UnivMon" << endl;
            update1(tuple_pairs);
        } else {
            cout << "updateing Non-UnivMon" << endl;
            update2(tuple_pairs);
        }

        auto end   = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        double secs = double(duration.count()) * microseconds::period::num / microseconds::period::den;
//        return double(tuple_pairs.size()) / secs / 1000000.0;
        return secs;
    }



}; // end of namespace

#endif //CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORCOMBINED_HPP
