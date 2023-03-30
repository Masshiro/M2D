//
// Created by Masshiro on 2022/9/21.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "../../src/hierarchy/UnivMonPro.hpp"
#include "gtest/gtest.h"
#include "fmt/core.h"

using namespace std;

const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
const std::string DATA_DIR = "../../data/caida/";

const uint32_t TOP_K = 100;
const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;//碰撞机率 8192
const uint32_t STAGE_COL = 1024; //单流精度 2048
const uint32_t LEVELS = 12;
const int REPEAT_TIMES = 1;
const string moments_names[7] = {"L1", "Le", "L2", "L3", "L4", "L5", "L6"};
std::random_device rd;

vector<pair<double, double> > test_one_file(int file_index, string data_path, string truth_path) {
    cout << " Processing file " << file_index << std::endl;
    vector<pair<double, double>> result(7, make_pair(0.0, 0.0));
    hierarchy::UnivMonSketchPro<On_vHLL> sketchPro(TOP_K, LEVELS, STAGE_NUM, STAGE_ROW, STAGE_COL, rd());

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

    for (int i = 0; i < REPEAT_TIMES; ++i) {
        infs.clear();
        infs.seekg(0);

        while (infs.good()) {
            infs >> src_ip >> dst_ip;
            if (infs.eof()) {
                break;
            }
            sketchPro.update(src_ip.c_str(), src_ip.length(), dst_ip.c_str(), dst_ip.length());
        }
    }

    true_infs.clear();
    true_infs.seekg(0);
    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof()) {
            break;
        }

        result[0].second += true_card;
        result[1].second += true_card * log(true_card);
        result[2].second += pow(true_card, 2);
        result[3].second += pow(true_card, 3);
        result[4].second += pow(true_card, 4);
        result[5].second += pow(true_card, 5);
        result[6].second += pow(true_card, 6);
    }

    result[0].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 1);
    result[1].first = sketchPro.calculate_moment_entropy(hierarchy::G_entropy);
    result[2].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 2);
    result[3].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 3);
    result[4].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 4);
    result[5].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 5);
    result[6].first = sketchPro.calculate_moment_power(hierarchy::G_sum, 6);
    return result;
}

TEST(UnivMonPro, basic) {
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
                 "\t relative error:\t" << abs(x_moment.first - x_moment.second)/x_moment.second << endl;
        }
        cout << '\n';
    }
}