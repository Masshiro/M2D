//
// Created by Masshiro on 2023/3/9.
//

#ifndef CARDINALITYMOMENTESTIMATOR_MAPIMPLFILTER_HPP
#define CARDINALITYMOMENTESTIMATOR_MAPIMPLFILTER_HPP

#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <queue>
#include <string>
#include <bitset>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

#include "../utils/HierarchyUtils.hpp"

using std::pair;
using std::vector;
using std::map, std::multimap;
using std::unordered_set, std::unordered_map;

namespace hierarchy {

    template<class KEY>
    class MapImplFilter {
    private:
        int top_k_num;
        int level_num;
        vector<multimap<double, KEY>> minheaps;
        vector<unordered_set<KEY>> flowIDs;
        int * element_num;
        int * hit_num;

    public:
        MapImplFilter(int top_k, int levels) {
            top_k_num = top_k;
            level_num = levels;
            minheaps = vector<multimap<double, KEY>>(level_num);
            flowIDs = vector<unordered_set<KEY>>(level_num);
            element_num = new int [level_num];
            hit_num = new int [level_num];

            for (int i = 0; i < level_num; ++i) {
                element_num[i] = 0;
                hit_num[i] = 0;
            }
        }

        ~MapImplFilter() {
            delete[] element_num;
            delete[] hit_num;
        }

        bool check_membership_at_level_k(KEY ID, int k) {
            if (flowIDs[k].count(ID) == 0) return false;
            else return true;
        }

        bool insert_element_at_level_k(KEY ID, double card, int k) {
            ++hit_num[k];
            //===================================================
            //      Condition 1: the hash has been inserted
            //===================================================
            if (check_membership_at_level_k(ID, k)) {
                for (auto it: minheaps[k]) {
                    if (it.second == ID) {
                        if (it.first < card) {
                            minheaps[k].erase(it.first);
                            minheaps[k].insert(std::make_pair(card, ID));
                        }
                        break;
                    }
                }
                return true;
            }

            //===================================================
            //      Condition 2: the hash is new here
            //===================================================
            /*      1). minheap at level k is not full yet      */
            if (minheaps[k].size() < top_k_num) {
                minheaps[k].insert(std::make_pair(card, ID));
                flowIDs[k].insert(ID);
                return true;
            }

            /*      2). minheap at level k is full now          */
            if (minheaps[k].size() >= top_k_num) {
                // item with smaller cardinality should be kicked out
                if (minheaps[k].begin()->first < card) {
                    flowIDs[k].erase(minheaps[k].begin()->second);
                    flowIDs[k].insert(ID);
                    minheaps[k].insert(std::make_pair(card, ID));
                    minheaps[k].erase(minheaps[k].begin()->first);
                    return true;
                }
            }
            return false;
        }

        vector<pair<double, double>> get_card_range_for_levels() {
            vector<pair<double, double>> res;
            for (int k = 0; k < level_num; ++k) {
                auto min_card = minheaps[k].begin()->first;
                auto max_card = (--minheaps[k].end())->first;
                res.emplace_back(std::make_pair(min_card, max_card));
            }
            return res;
        }

        void display_heap_at_level_k(int k, int base=10) {
            int cnt = 0;
            for (auto it: minheaps[k]) {
                std::cout << "<" << it.second << ", " << it.first << "> ";
                if (++cnt % base == 0) std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        vector<pair<KEY, double>> get_elements_at_level_k(int k) {
            vector<pair<KEY, double>> res;
            for (auto p: minheaps[k]) {
                res.emplace_back(std::make_pair(p.second, p.first));
            }
            return res;
        }

        unordered_map<double, double> get_heaps_card_freq_info() {
            unordered_map<double, double> res;
            for (int k = 0; k < level_num; ++k) {
                for (auto p: minheaps[k]) {
                    ++res[p.first];
                }
            }
            return res;
        }

        unordered_map<double, double> get_heaps_card_freq_info_at_level_k(int k) {
            unordered_map<double, double> res;
            for (auto p: minheaps[k]) {
                ++res[p.first];
            }
            return res;
        }



    };
} // the end of namespace


#endif //CARDINALITYMOMENTESTIMATOR_MAPIMPLFILTER_HPP
