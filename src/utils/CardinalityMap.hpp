#ifndef CARDINALITY_MAP_HPP
#define CARDINALITY_MAP_HPP

#include "spdlog/spdlog.h"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

inline double get_r_error(double true_card, double est_card) {
    return (est_card - true_card) / true_card;
}

inline double iter_avg(int before_num, double before_avg_val, double new_val,
                       int new_num = 1) {
    return before_avg_val * ((double)before_num / (before_num + new_num)) +
           new_num * new_val / (before_num + new_num);
}

class card_map_val {
public:
    long long num;
    double r_err;
    double r_err_pow2;
    card_map_val(int num, double r_err)
        : num(num), r_err(r_err), r_err_pow2(r_err * r_err) {}

    card_map_val(int num, double r_err, double r_err_pow2)
        : num(num), r_err(r_err), r_err_pow2(r_err_pow2) {}

    void add(double r_err) {
        double r_err_pow2 = r_err * r_err;
        this->r_err = iter_avg(this->num, this->r_err, r_err);
        this->r_err_pow2 = iter_avg(this->num, this->r_err_pow2, r_err_pow2);
        this->num++;
    }

    void merge(card_map_val &input) {
        this->r_err = iter_avg(this->num, this->r_err, input.r_err, input.num);
        this->r_err_pow2 =
            iter_avg(this->num, this->r_err_pow2, input.r_err_pow2, input.num);
        this->num += input.num;
    }
};

static std::shared_ptr<card_map_val> create_card_map_val(int num,
                                                         double r_err) {
    std::shared_ptr<card_map_val> ptr_card_map_val =
        std::make_shared<card_map_val>(num, r_err);
    return ptr_card_map_val;
}

static std::shared_ptr<card_map_val> create_card_map_val(int num, double r_err,
                                                         double r_err_pow2) {
    std::shared_ptr<card_map_val> ptr_card_map_val =
        std::make_shared<card_map_val>(num, r_err, r_err_pow2);
    return ptr_card_map_val;
}

class CardinalityMap {
private:
    std::mutex mtx;
    std::map<int, std::shared_ptr<card_map_val>> data;
    int thres_hold;

public:
    CardinalityMap(int thres_hold = 10) : thres_hold(thres_hold) {}
    void add(int card, double r_err) {
        if (card < thres_hold) {
            return;
        }
        try {
            std::lock_guard<std::mutex> lck(mtx);
            if (data.count(card) == 0) {
                data[card] = create_card_map_val(1, r_err);
            } else {
                data[card]->add(r_err);
            }
        } catch (std::logic_error &) {
            spdlog::error("[exception caught]");
        }
    }

    void read_from_file(std::string input_path) {
        std::ifstream infs(input_path);
        int card_key = 0;
        int num = 0;
        double r_err = 0;
        double r_err_pow2 = 0;
        while (infs.good()) {
            infs >> card_key >> num >> r_err >> r_err_pow2;
            if (infs.eof()) {
                break;
            }
            auto cmv = create_card_map_val(num, r_err, r_err_pow2);
            if (data.count(card_key) == 0) {
                data[card_key] = cmv;
            } else {
                data[card_key]->merge(*cmv);
            }
        }
    }

    void save_to_file(std::string output_path) {
        std::ofstream ofst(output_path);
        for (auto map_it = data.cbegin(); map_it != data.cend(); map_it++) {
            int card_key = map_it->first;
            auto val = map_it->second;
            /*
                card_key    <--->   true cardinality
                num         <--->   sample size(actual sample size * REPEAT_TIME)
                r_err       <--->   average relative error
                r_err_pow2  <--->   r_err * r_err
            */
            ofst << card_key << '\t' << val->num << '\t' << val->r_err << '\t'
                 << val->r_err_pow2 << std::endl;
        }
        ofst.close();
    }

    void show() {
        for (auto map_it = data.cbegin(); map_it != data.cend(); map_it++) {
            int card_key = map_it->first;
            auto val = map_it->second;
            /*
                card_key    <--->   true cardinality
                num         <--->   sample size(actual sample size * REPEAT_TIME)
                r_err       <--->   average relative error
                r_err_pow2  <--->   r_err * r_err
            */
            std::cout << card_key << '\t' << val->num << '\t' << val->r_err << '\t'
                 << val->r_err_pow2 << std::endl;
        }
    }

    void merge(CardinalityMap &input) {
        std::lock_guard<std::mutex> lck(mtx);
        for (auto map_it = input.data.begin(); map_it != input.data.end();
             map_it++) {
            int card = map_it->first;
            if (card < thres_hold) {
                continue;
            }
            card_map_val input_val = *(map_it->second);
            if (this->data.count(map_it->first) == 0) {
                data[card] = std::make_shared<card_map_val>(input_val);
            } else {
                data[card]->merge(input_val);
            }
        }
    }
};

#endif
