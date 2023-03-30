//
// Created by Masshiro on 2022/9/21.
//

#ifndef CARDINALITYMOMENTESTIMATOR_MINHEAPFILTER_HPP
#define CARDINALITYMOMENTESTIMATOR_MINHEAPFILTER_HPP
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <queue>
#include <string>
#include <bitset>
#include <iostream>

#include "../utils/HierarchyUtils.hpp"

namespace hierarchy{

//    template<class KEY, class VAL>
//    struct CompareByValue{
//        bool operator() (const std::pair<KEY, VAL>& p1, const std::pair<KEY, VAL>& p2){
//            return p1.second > p2.second;
//        }
//    };

    template<class KEY>
    class MinHeapFilter {
    private:
        uint32_t TOP_K;
        uint32_t LEVEL_NUM;
        std::pair<KEY, double> ** minheaps;
        std::vector<std::pair<KEY, double>> * sorted_minheaps;
        int * MIN_INDICES;
        int * ELEMENTS_NUM;
        int * HIT_NUM;

    public:
        MinHeapFilter(uint32_t top_k, uint32_t level_num){
            TOP_K = top_k;
            LEVEL_NUM = level_num;
            minheaps = new std::pair<KEY, double>*[LEVEL_NUM];
            sorted_minheaps = new std::vector<std::pair<KEY, double>> [LEVEL_NUM];
            MIN_INDICES = new int [LEVEL_NUM];
            ELEMENTS_NUM = new int [LEVEL_NUM];
            HIT_NUM = new int [LEVEL_NUM];

            for (int i = 0; i < LEVEL_NUM; ++i) {
                minheaps[i] = new std::pair<KEY, double>[TOP_K]; // {std::make_pair(0, 0.0)};
                MIN_INDICES[i] = 0;
                ELEMENTS_NUM[i] = 0;
                HIT_NUM[i] = 0;
            }
        }

        ~MinHeapFilter(){
            for (int i = 0; i < LEVEL_NUM; ++i) {
                delete[] minheaps[i];
            }
            delete[] MIN_INDICES;
            delete[] ELEMENTS_NUM;
        }

        int check_membership_at_level_k (KEY ID, int k){
            for (int i = 0; i < ELEMENTS_NUM[k]; ++i) {
                if (minheaps[k][i].first == ID) {
                    return i;
                } else {
                    continue;
                }
            }
            return -1;
        }

        void set_minval_index_at_level_k(int k){
            int min_index = 0;
            for (int i = 0; i < ELEMENTS_NUM[k]; ++i) {
                if (minheaps[k][i].second < minheaps[k][min_index].second)
                    min_index = i;
            }
            MIN_INDICES[k] = min_index;
        }

        bool insert_element_at_level_k (KEY ID, double card, int k){
            HIT_NUM[k] ++;
            //===================================================
            //      Condition 1: the hash has been inserted
            //===================================================
            int position = check_membership_at_level_k(ID, k);
            if (position != -1){
                if (minheaps[k][position].second < card) {
                    minheaps[k][position].second = card;
                    set_minval_index_at_level_k(k);
                }
//            minheaps[k][position].second = card;
//            set_minval_index_at_level_k(k);
                return true;
            }

            //===================================================
            //      Condition 2: the hash is new here
            //===================================================
            /*      1). minheap at level k is not full yet      */
            if (ELEMENTS_NUM[k] < TOP_K) {
                minheaps[k][ELEMENTS_NUM[k]].first = ID;
                minheaps[k][ELEMENTS_NUM[k]].second = card;
                ELEMENTS_NUM[k] ++;
                set_minval_index_at_level_k(k);
                return true;
            }

            /*      2). minheap at level k is full now          */
            if (ELEMENTS_NUM[k] == TOP_K) {
                // item with smaller cardinality should be kicked out
                if (minheaps[k][MIN_INDICES[k]].second < card){
                    minheaps[k][MIN_INDICES[k]].first = ID;
                    minheaps[k][MIN_INDICES[k]].second = card;
                    set_minval_index_at_level_k(k);
                    return true;
                } else {
                    return false;
                }
            }
        }

        void multiply_heap_results_by_n (double n) {
            for (int i = 0; i < LEVEL_NUM; ++i) {
                for (int j = 0; j < ELEMENTS_NUM[i]; ++j) {
                    minheaps[i][j].second *= n;
                }
            }
        }

        void print_list_at_level_k(int k){
            std::bitset<32> hashvalue;
            std::cout << " \nLevel " << k << "\tPacket count: " << HIT_NUM[k] << "\tElement num: " << ELEMENTS_NUM[k] << std::endl;
            for (int i = 0; i < ELEMENTS_NUM[k]; ++i) {
                std::cout <<minheaps[k][i].first << " --> "<< minheaps[k][i].second << '\t';
                if (i % (ELEMENTS_NUM[k] / 4) == 0){
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::vector<std::pair<KEY, double>> * get_sorted_heaps() {
            for (int i = 0; i < LEVEL_NUM; ++i) {
                for (int j = 0; j < ELEMENTS_NUM[i]; ++j) {
                    sorted_minheaps[i].emplace_back(minheaps[i][j]);
                }
                std::sort(sorted_minheaps[i].begin(), sorted_minheaps[i].end(), hierarchy::CompareByValue<KEY, double>());
            }
            return sorted_minheaps;
        }

        std::vector<std::pair<double, double>> get_card_range_for_levels() {
            std::vector<std::pair<double, double>> res;
            for (int i = 0; i < LEVEL_NUM; ++i) {
                std::vector<double> sorted_single_level_cards;
                for (int j = 0; j < ELEMENTS_NUM[i]; ++j) {
                    sorted_single_level_cards.emplace_back(minheaps[i][j].second);
                }
                std::sort(sorted_single_level_cards.begin(), sorted_single_level_cards.end());
                res.emplace_back(std::make_pair(sorted_single_level_cards[0], sorted_single_level_cards.back()));
            }
            return res;
        }

        std::pair<KEY, double>* get_minheap_at_level_k(int k) {
            return minheaps[k];
        }

        int get_minval_index_at_level_k(int k){
            return MIN_INDICES[k];
        }

        int get_storage_condition_at_level_k (int k){
            return ELEMENTS_NUM[k];
        }

        std::pair<KEY, double>* get_elements_at_level_k(int k){
            return minheaps[k];
        }

        int get_level_num(){
            return LEVEL_NUM;
        }


    };

}   // the end of namespace

#endif //CARDINALITYMOMENTESTIMATOR_MINHEAPFILTER_HPP
