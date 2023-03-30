//
// Created by Masshiro on 2023/3/14.
//

#ifndef CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORPRO_HPP
#define CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORPRO_HPP

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

#include "../sketch/On_vHLL.hpp"
#include "../sketch/Ton_vHLL.hpp"
#include "../utils/HierarchyUtils.hpp"
#include "../filter/MinheapFilter.hpp"
#include "../filter/MapImplFilter.hpp"
#include "fmt/core.h"

using std::string, std::vector, std::map, std::bitset;
using std::unordered_map, std::unordered_set;
using std::cout, std::endl;
using std::ifstream, std::ofstream;

const int LOWER_BOUND = 10;

namespace hierarchy {

    template<class SKETCH>
    class MomentEstimatorPro {
    private:
        SKETCH **sketches;
        hierarchy::MapImplFilter<string> filters;
        unordered_map<string, unordered_set<string>> hash_table;

        int *count_level_pkts;

        uint32_t m_seed;
        uint32_t m_level_count;
        uint32_t m_filter_size;
        uint32_t m_sketch_rows;
        double m_prog_rate;
        bool m_hash_table_enable;
        vector<unordered_set<string>> recorded_flow_ids;
        unordered_map<string, double> true_card_dict;

    public:
        MomentEstimatorPro(uint32_t top_k, uint32_t level_num,
                        uint32_t sketch_stages, uint32_t sketch_rows, uint32_t sketch_cols,
                        uint32_t master_seed, string filename, double prog_rate = 2, bool enable_hash_table=true) : filters(top_k,level_num) {
            m_filter_size = top_k;
            m_level_count = level_num;
            m_seed = master_seed;
            m_sketch_rows = sketch_rows;
            m_prog_rate = prog_rate;
            m_hash_table_enable = enable_hash_table;

            count_level_pkts = new int[m_level_count + 1];
            for (int i = 0; i < m_level_count + 1; ++i) {
                count_level_pkts[i] = 0;
            }
            sketches = new SKETCH *[m_level_count];

            if (level_num > 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_int_distribution<uint32_t> dist(1, 1e6);
                for (uint32_t j = 0; j != m_level_count; ++j) {
                    uint32_t seed = dist(mt);
                    sketches[j] = new SKETCH(sketch_stages, sketch_rows / pow(prog_rate, j), sketch_cols, seed);
                }
            }

            //// construct true cardinality search dictionary
            ifstream true_card_file(filename);
            if (!true_card_file) {
                spdlog::error("open {} fail", filename);
                true_card_file.close();
            }
            int true_card = 0.0;
            string flow_id;
            true_card_file.clear(); true_card_file.seekg(0);
            while(true_card_file.good()) {
                if (true_card_file.eof()) break;
                true_card_file >> flow_id >> true_card;
                true_card_dict[flow_id] = true_card;
            }

            recorded_flow_ids = vector<unordered_set<string>> (m_level_count);
        }

        ~MomentEstimatorPro() {
            for (uint32_t i = 0; i < m_level_count; ++i) {
                delete sketches[i];
            }
            delete sketches;
            delete count_level_pkts;
        }

        int memory_usage_int_bits() {
            int bytes_used = 0;
            int bits_used = 0;
            // sketch and filter
            for (int i = 0; i < m_level_count; ++i) {
                bits_used += sketches[i]->memory_usage_in_bits();
                bytes_used += (sizeof(int) + sizeof(double)) * m_filter_size;
            }
            // hash table
            if (m_hash_table_enable) {
                bytes_used += sizeof(int) * 2 * hash_table.size();
            }

            return bytes_used * CHAR_BIT + bits_used;
        }

        bool check_flowid_in_filter_at_level_k(string flow_id, int k) {
            return filters.check_membership_at_level_k(flow_id, k);
        }

        int get_sample_level(uint32_t hash_val) {
            bitset<32> bi_hash_val = hash_val;
            int lead_cnt = 0;
            for (; lead_cnt < 32; ++lead_cnt) {
                if (bi_hash_val[lead_cnt] == 1) break;
            }
            ++lead_cnt;
            if (lead_cnt % (int(m_prog_rate) / 2) != 0) return -1;
            int idx = lead_cnt / (m_prog_rate / 2);
            return std::min(idx, int(m_level_count));
        }

        vector<pair<double, double>> get_card_ranges() {
            return filters.get_card_range_for_levels();
        }

        void display_spreader_ranges() {
            auto ranges = filters.get_card_range_for_levels();
            for (auto level_range: ranges) {
                cout << level_range.first << "\t" << level_range.second << endl;
            }
            cout << endl;
        }

        void update_sketches(string, string);

        void build_filters();

        void update_filters(string, double, int);

        void update_hash_table(string, string);

        void update_sketch_at_level_k(string, string, uint32_t);

        void display_pkts_each_level();

        double calculate_moment_power(double(*f)(double x, uint8_t p), uint8_t);

    };

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::update_hash_table(string flow_id, string element_id) {
        ++count_level_pkts[m_level_count];
        hash_table[flow_id].insert(element_id);
    }

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::update_filters(string ID, double card, int k) {
        bool flag = filters.insert_element_at_level_k(ID, card, k);
        if (flag) {
            if (k == 0) return;
            else {
                for (int i = k - 1; i >= 0; --i) {
                    filters.insert_element_at_level_k(ID, card, i);
                }
            }
        }
    }

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::update_sketch_at_level_k(string flow_id, string element_id, uint32_t k) {
        ++count_level_pkts[k];
        recorded_flow_ids[k].insert(flow_id);
        sketches[k]->offerFlow(flow_id.c_str(), flow_id.length(), element_id.c_str(), element_id.length());
    }

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::update_sketches(string flow_id, string element_id) {
        if (m_level_count == 0) {
            if (m_hash_table_enable) {
                update_hash_table(flow_id, element_id);
            } else {
                cout << "Check the level setting" << endl;
                return;
            }
        }

        XXHash32 hash32(m_seed);
        hash32.add(flow_id.c_str(), flow_id.length());
        uint32_t flow_id_hash = hash32.hash();
        uint32_t j_star = get_sample_level(flow_id_hash);

        update_sketch_at_level_k(flow_id, element_id, 0);

        if (j_star == -1) return;

        if (j_star == m_level_count) {
            if (m_hash_table_enable) {
                update_hash_table(flow_id, element_id);
            } else {
                return;
            }
        } else if (j_star != 0) {
            update_sketch_at_level_k(flow_id, element_id, j_star);
        }
    }

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::build_filters() {
        //// start with hash table layer
        for (auto flow: hash_table) {
            update_filters(flow.first, flow.second.size(), m_level_count-1);
        }

        //// other layers
        for (int k = m_level_count-1; k >= 0; --k) {
            for (auto flow_id: recorded_flow_ids[k]) {
                if (true_card_dict[flow_id] < LOWER_BOUND) continue;
                else {
                    double esti_val = sketches[k]->decodeFlow(flow_id.c_str(), flow_id.length());
                    if (esti_val > 0.0) {
                        update_filters(flow_id, esti_val, k);
                    }
                }
            }
        }
    }

    template<class SKETCH>
    void MomentEstimatorPro<SKETCH>::display_pkts_each_level() {
        int base = count_level_pkts[0];
        for (int i = 0; i < m_level_count; ++i) {
            std::cout << "\t -> level " << i << ": " << count_level_pkts[i] << "\tpercentage: "
                      << double(count_level_pkts[i]) / double(base) << "(" << 1 / pow(m_prog_rate, i) << ")"
                      << std::endl;
        }
        if (m_hash_table_enable) {
            std::cout << "\t -> hash table: " << count_level_pkts[m_level_count] << "\tpercentage: "
                      << double(count_level_pkts[m_level_count]) / double(base) << "("
                      << 1 / pow(m_prog_rate, m_level_count) << ")"
                      << std::endl;
        }
    }

    template<class SKETCH>
    double MomentEstimatorPro<SKETCH>::calculate_moment_power(double (*f)(double, uint8_t), uint8_t power) {
        if (power == 1 && m_level_count != 0) {
            return sketches[0]->getNHat();
        }
        double moment = 0.0;

        if (m_hash_table_enable) {
            for (auto flow: hash_table) {
                moment += f(flow.second.size(), power);
            }
            if (m_level_count == 0) return moment;
        }

        for (int k = m_level_count - 1; k >= 0; --k) {
            moment *= m_prog_rate;
            auto heavy_hitters = filters.get_elements_at_level_k(k);

            for (auto hh: heavy_hitters) {
                string flow_id = hh.first;
                XXHash32 hash32(m_seed);
                hash32.add(flow_id.c_str(), flow_id.length());
                int deepest_level = get_sample_level(hash32.hash());

                bool hf = 0;
                if (k < deepest_level) { //k < deepest_level
                    hf = 1;
                }
                if (hh.second != 0.0) {
                    moment += f(hh.second, power) * (1 - m_prog_rate * hf);
                }
            }
        }

        return moment;
    }
};

#endif //CARDINALITYMOMENTESTIMATOR_MOMENTESTIMATORPRO_HPP
