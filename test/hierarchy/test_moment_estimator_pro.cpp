//
// Created by Masshiro on 2023/3/14.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "../../src/hierarchy/MomentEstimatorPro.hpp"
#include "gtest/gtest.h"
#include "fmt/core.h"

using namespace std;

const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
const std::string DATA_DIR = "../../data/caida/";

const int sampling_prob = 2;
const uint32_t LEVELS = 5;

const uint32_t TOP_K = 100;
const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;//  w, 碰撞机率 8192
const uint32_t STAGE_COL = 1024; // d, 单流精度 2048
const int REPEAT_TIMES = 1;
const string moments_names[7] = {"L1", "L3", "L5", "L7", "L10", "L12", "L15"};
std::random_device rd;
vector<unordered_map<string, double>> true_card_dict_files(5);

void load_true_cards() {
    int file_idx = 0;
    string readline;
    ifstream ifls(INPUT_FILE_LIST);
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof()) {
            break;
        }
        string truth_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);
        ifstream true_infs(truth_path);
        int true_card = 0;
        string src_ip;
        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof()) {
                break;
            }
            true_card_dict_files[file_idx][src_ip] = true_card;
        }
        ++file_idx;
    }
}

vector<pair<double, double> > test_one_file(int file_index, string data_path, string truth_path) {
    cout << " Processing file " << file_index << std::endl;
    vector<pair<double, double>> result(7, make_pair(0.0, 0.0));
//    hierarchy::MomentEstimatorPro<On_vHLL> sketchPro(TOP_K, LEVELS, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), truth_path,sampling_prob);
    hierarchy::MomentEstimatorPro<Ton_vHLL> sketchPro(TOP_K, LEVELS, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), truth_path,sampling_prob);

    ifstream true_infs(truth_path);
    ifstream infs(data_path);

    if (!infs) {
        spdlog::error("open {} fail", data_path);
        infs.close();
        true_infs.close();
    }
    int true_card = 0;
    string src_ip;
    string dst_ip;
    int pkt_cnt = 0;

    for (int i = 0; i < REPEAT_TIMES; ++i) {
        infs.clear();
        infs.seekg(0);

        while (infs.good()) {
            ++pkt_cnt;
            infs >> src_ip >> dst_ip;
            if (infs.eof()) {
                break;
            }
            sketchPro.update_sketches(src_ip, dst_ip);
        }
        sketchPro.build_filters();
    }

//    sketchPro.display_spreader_ranges();

    cout << "memory cost: " << sketchPro.memory_usage_int_bits() / 8388608.0 << " MB\t packets: " << pkt_cnt << endl;
//    cout << "hash table size: " << sketchPro.get_hash_table_size() << endl;
    sketchPro.display_pkts_each_level();
//    sketchPro.diplay_filter_cards_ranges();

    true_infs.clear();
    true_infs.seekg(0);
    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof()) {
            break;
        }

        result[0].second += true_card;
        result[1].second += pow(true_card, 3);
        result[2].second += pow(true_card, 5);
        result[3].second += pow(true_card, 7);
        result[4].second += pow(true_card, 10);
        result[5].second += pow(true_card, 12);
        result[6].second += pow(true_card, 15);
    }

    result[0].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 1);
    result[1].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 3);
    result[2].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 5);
    result[3].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 7);
    result[4].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 10);
    result[5].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 12);
    result[6].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 15);
    return result;
}

TEST(MomentEstimator, basic) {
//    load_true_cards();

    int file_index = 0;
    std::string readline;
    vector<vector<pair<double, double>>> results;

    std::ifstream ifls(INPUT_FILE_LIST);
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof()) {
            break;
        }
        std::string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        std::string truth_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);
        results.emplace_back(test_one_file(++file_index, input_data_path, truth_data_path));
    }

    std::cout << std::endl;
    for (auto one_result: results) {
        int name_index = 0;
        for (auto x_moment: one_result) {
            cout << moments_names[name_index++] << '\t';
            cout << "estimated:\t" << x_moment.first << "\t\ttruth:\t" << x_moment.second <<
                 "\t relative error:\t" << (x_moment.first - x_moment.second)/x_moment.second << endl;
        }
        cout << '\n';
    }
}