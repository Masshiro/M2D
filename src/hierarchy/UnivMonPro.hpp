//
// Created by Masshiro on 2022/9/21.
//

#ifndef CARDINALITYMOMENTESTIMATOR_UNIVMONPRO_HPP
#define CARDINALITYMOMENTESTIMATOR_UNIVMONPRO_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <type_traits>
#include <fstream>
#include <string>
#include <vector>

#include "../sketch/On_vHLL.hpp"
#include "../filter/MinheapFilter.hpp"
#include "../utils/HierarchyUtils.hpp"
#include "fmt/core.h"

namespace hierarchy{
    template<class SKETCH>
    class UnivMonSketchPro{
    private:
        SKETCH ** sketch;
        hierarchy::MinHeapFilter<uint32_t> filter;
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, bool> > hash_table;

        int * count_level_pkts;

        uint32_t m_seed;
        uint32_t m_level_count;
        uint32_t m_filter_size;
        uint32_t m_sketch_rows;
        std::vector<double> moment_for_each_level;

    public:
        UnivMonSketchPro(uint32_t top_k, uint32_t level_num, uint32_t sketch_stages,
                         uint32_t sketch_rows, uint32_t sketch_cols, uint32_t sketch_seed) : filter(top_k, level_num){
            m_filter_size = top_k;
            m_level_count = level_num;
//            srand(sketch_seed);
            m_seed = sketch_seed;
            m_sketch_rows = sketch_rows;

            sketch = new On_vHLL * [m_level_count];
            for (uint32_t j = 0; j != m_level_count; j++){
                sketch[j] = new On_vHLL(sketch_stages, sketch_rows, sketch_cols, sketch_seed);
            }

            count_level_pkts = new int [m_level_count];
            for (int i = 0; i < m_level_count; i++){
                count_level_pkts[i] = 0;
            }
        }

        ~UnivMonSketchPro(){
            for (uint32_t i = 0; i != m_level_count; i++){
                delete sketch[i];
            }
            delete sketch;
            delete count_level_pkts;
        }

        static inline uint32_t get_leading_ones(uint32_t a, uint32_t max_len){
            uint32_t mask = 1;
            for (int i = 0; i != max_len; ++i){
                if ((a & mask) != 0){
                    mask = mask<<1;
                } else {
                    return i;
                }
            }
            return max_len;
        }

        uint32_t get_level_count();

        void update_sketch_at_level_k(const void *, uint64_t, const void *, uint64_t, uint32_t);

        bool update_filter_at_level_k(uint32_t, double, uint32_t);

        void update(const void*, uint64_t, const void*, uint64_t);

        double calculate_moment_power(double(*f)(double x, uint8_t p), uint8_t);

        double calculate_moment_entropy(double(*f)(double x));

        void display_filters();


        void export_filters_info(std::string, std::string);

    };

    template<class SKETCH>
    uint32_t UnivMonSketchPro<SKETCH>::get_level_count() {
        return m_level_count;
    }

    template<class SKETCH>
    void UnivMonSketchPro<SKETCH>::update_sketch_at_level_k(const void *ptr_flow_id, uint64_t flow_id_len,
                                                            const void *ptr_element_id, uint64_t element_id_len, uint32_t k) {
        sketch[k]->offerFlow(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len);
    }

    template<class SKETCH>
    bool UnivMonSketchPro<SKETCH>::update_filter_at_level_k(uint32_t ID, double cardinality, uint32_t k) {
        return filter.insert_element_at_level_k(ID, cardinality, k);
    }

    template<class SKETCH>
    void UnivMonSketchPro<SKETCH>::update(const void *ptr_flow_id, uint64_t flow_id_len,
                                          const void * ptr_element_id, uint64_t element_id_len) {
        XXHash32 hash32(m_seed);
        hash32.add(ptr_flow_id, flow_id_len);
        uint32_t rnd = hash32.hash();
        uint32_t sample_levels = get_leading_ones(rnd, m_level_count -1);
        //  If the value of 'sample_levels' equals to 0,
        //  the flow will still be sampled into level 0.
//    std::cout << "Inserting flow: \t Flow ID: "<<rnd<<"\t Sample levels: "<<sample_levels<<std::endl;

        for (int k = sample_levels; k != -1; k--){
            update_sketch_at_level_k(ptr_flow_id, flow_id_len, ptr_element_id, element_id_len, k);
            double estimate_value = sketch[k]->decodeFlow(ptr_flow_id, flow_id_len);

            if (estimate_value > 0.0) {
                update_filter_at_level_k(rnd, estimate_value, k);
            } else {
                continue ;
            }
//        std::cout << "Updating level "<< k << " \t estimated cardinality is "<<estimate_value<<std::endl;
        }
    }


    template<class SKETCH>
    double UnivMonSketchPro<SKETCH>::calculate_moment_power(double (*f)(double, uint8_t), uint8_t power){
        if (power == 1 && m_level_count != 0) {
            return sketch[0]->getNhatSlow();
        }

        double below_level_moment = 0.0;
        std::pair<uint32_t, double>* ptr_level_k;

        for (auto iter: hash_table) {
            uint32_t sample_levels = get_leading_ones(iter.first, m_level_count);
            int hf = 0;
            if (m_level_count < sample_levels) {
                hf = 1;
            }

            below_level_moment = below_level_moment + f(iter.second.size(), power) * (1-2*hf);
        }

        for (int i = m_level_count - 1; i != -1; --i) {
            below_level_moment *= 2;
            ptr_level_k = filter.get_elements_at_level_k(i);
            for (int j = 0; j < filter.get_storage_condition_at_level_k(i); ++j) {
                uint32_t sample_levels = get_leading_ones(ptr_level_k[j].first, m_level_count);

                int hf =  0;

                if (i < sample_levels){
                    hf = 1;
                }

                if (ptr_level_k[j].second != 0.0){
                    below_level_moment = below_level_moment + f(ptr_level_k[j].second, power) * (1-2*hf);
                }
            }
        }

        return below_level_moment;
    }

    template<class SKETCH>
    double UnivMonSketchPro<SKETCH>::calculate_moment_entropy(double (*f)(double)) {
        double below_level_moment = 0.0;
        std::pair<uint32_t, double>* ptr_level_k;

        for (int i = m_level_count - 1; i != -1; --i) {
            below_level_moment *= 2;
            ptr_level_k = filter.get_elements_at_level_k(i);
            for (int j = 0; j < filter.get_storage_condition_at_level_k(i); ++j) {
                uint32_t sample_levels = get_leading_ones(ptr_level_k[j].first, m_level_count -1);

                int hf =  0;

                if (i < sample_levels){
                    hf = 1;
                }

                if (ptr_level_k[j].second != 0.0){
                    below_level_moment = below_level_moment + f(ptr_level_k[j].second) * (1-2*hf);
                }
            }
        }
        return below_level_moment;
    }

    template<class SKETCH>
    void UnivMonSketchPro<SKETCH>::display_filters() {
//    filter.print_storage_condition_for_all_levels();
        for (int i = 0; i < filter.get_level_num(); i++){
//        std::cout << "---------- Level "<<i<<" ----------  Storage condition: "<<filter.get_storage_condition_at_level_k(i)<<std::endl;
            filter.print_list_at_level_k(i);
        }
    }

    template<class SKETCH>
    void UnivMonSketchPro<SKETCH>::export_filters_info(std::string save_path, std::string filename) {
        std::ofstream outf(save_path + filename);

        std::pair<uint32_t, double>* ptr_level_k;
        for (int i = 0; i < filter.get_level_num(); i++){
            ptr_level_k = filter.get_elements_at_level_k(i);
            outf << "---------- Level "<<i<<" ----------  Storage condition: "<<filter.get_storage_condition_at_level_k(i)<<std::endl;
            for (int j = 0; j < filter.get_storage_condition_at_level_k(i); ++j) {
                outf << ptr_level_k[j].first << ' ' << ptr_level_k[j].second << '\n';
            }
            outf << std::endl;
        }
    }
}   //  the end of the namespace
#endif //CARDINALITYMOMENTESTIMATOR_UNIVMONPRO_HPP
