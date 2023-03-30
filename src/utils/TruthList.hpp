//
// Created by Masshiro on 2022/9/28.
//

#ifndef CARDINALITYMOMENTESTIMATOR_TRUTHLIST_HPP
#define CARDINALITYMOMENTESTIMATOR_TRUTHLIST_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <string>
#include <bitset>
#include <unordered_map>

#include "HierarchyUtils.hpp"

using namespace std;



template<class KEY>
class TruthList{
private:
    uint32_t level_num;                                          //  to support the multi-layer situation
    unordered_map<KEY, unordered_map<KEY, bool> >* hash_tables;  //  used while inserting arrival pkt
    vector<KEY>* flow_ids;                                       //  used to store flow IDs
    vector<pair<KEY, int> >* truth_lists;                        //  sorted top-k list
    bool lists_ready;

public:
    TruthList(uint32_t levels = 1) {
        lists_ready = false;
        level_num = levels;
        hash_tables = new unordered_map<KEY, unordered_map<KEY, bool> > [level_num];
        flow_ids = new vector<KEY> [level_num];
        truth_lists = new vector<pair<KEY, int> > [level_num];
    }

    ~TruthList() {
        delete [] truth_lists;
        delete [] hash_tables;
        delete [] flow_ids;
    }

    //  insert every arrival packet to the data structure
    void insert_element_at_level_k(KEY flow_id, KEY element_id, uint32_t k = 0) {
        auto position = hash_tables[k].find(flow_id);
        if (position == hash_tables[k].end()) {     //  this flow ID has not been inserted
            flow_ids[k].emplace_back(flow_id);
            hash_tables[k][flow_id].insert(std::make_pair(element_id, true));
        } else {                                    //  this flow ID has been inserted
            if (position->second.find(element_id) == position->second.end()) {
                //  the element is unique, we need to insert it
                position->second.insert(std::make_pair(element_id, true));
            }
        }
    }

    //  make truth set of top-k available, which is consisted by the contents of truth_lists
    void process_data(int card_lower_bound = 1) {
        for (int i = 0; i < level_num; ++i) {
            for (const auto& flow: hash_tables[i]) {
                if (flow.second.size() >= card_lower_bound) {
                    truth_lists[i].emplace_back(std::make_pair(flow.first, flow.second.size()));
                }
            }
            std::sort(truth_lists[i].begin(), truth_lists[i].end(), hierarchy::CompareByValue<KEY, int>());
        }
        lists_ready = true;
    }

    vector<pair<KEY, int> > get_elements_at_level_k(uint32_t k = 0) {
        return truth_lists[k];
    }

    int get_hashtable_size_at_level_k(uint32_t k = 0) {
        return hash_tables[k].size();
    }

    int get_value_with_id_at_level_k(KEY id, uint32_t k) {
        if (hash_tables[k].find(id) == hash_tables[k].end()) {
            return -1;
        }
        return hash_tables[k][id].size();
    }

    void print_truth_list_top_n_at_level_k(int n = 30, uint32_t k = 0){
        if (!lists_ready) {
            process_data();
        }
        std::cout << "\ntruth list at level " << k << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << truth_lists[k][i].first << " --> " << truth_lists[k][i].second << std::endl;
        }
        std::cout << std::endl;
    }

    std::pair<bool, int> check_one_flow_correctness_at_level_k(KEY flow_id, int truth, int k) {
        return std::make_pair(truth == hash_tables[k][flow_id].size(), hash_tables[k][flow_id].size());
    }

};

#endif //CARDINALITYMOMENTESTIMATOR_TRUTHLIST_HPP
