//
// Created by Masshiro on 2023/3/23.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "../../src/hierarchy/MomentEstimatorCombined.hpp"
#include "gtest/gtest.h"
#include "fmt/core.h"

using namespace std;

const int sys_name_idx = 0;
const std::string DATA_DIR = "../../data/caida/";
vector<string> sys_names = {"UnivMon", "LUS", "M2D"};

const int TOP_K = 100;
const int LEVEL_NUM = 6;
const int STAGE_NUM = 4;
const int STAGE_ROW = 4096;//  w, 碰撞机率 8192
const int STAGE_COL = 1024; // d, 单流精度 2048

std::random_device rd;


int main() {
    //// load tuples
    vector<pair<string, string>> tuples;
    string data_file_name = fmt::format("{}equinix-chicago.dirA.20160121-133300.UTC.anon.pcap.gz.uniq", DATA_DIR);
    ifstream data_file(data_file_name);
    if (!data_file) {
        spdlog::error("open {} fail", data_file_name);
        data_file.close();
    }
    data_file.clear();
    data_file.seekg(0);
    while (data_file.good()) {
        string src_ip, dst_ip;
        data_file >> src_ip >> dst_ip;
        if (data_file.eof()) {
            break;
        }
        tuples.emplace_back(make_pair(src_ip, dst_ip));
    }

    //// test for UnivMon
    hierarchy::MomentEstimatorCombined estimator1(TOP_K, LEVEL_NUM, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), 2, "UnivMon");
    double univmon_res = estimator1.update(tuples);
    cout << double(tuples.size()) / univmon_res / 1000000 << endl;

    //// test for LUS
    hierarchy::MomentEstimatorCombined estimator2(TOP_K, LEVEL_NUM, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), 2, "LUS");
    double lus_res = estimator2.update(tuples);
    cout << double(tuples.size()) / lus_res / 1000000 << endl;

    //// test for M2D (1/4)
    hierarchy::MomentEstimatorCombined estimator3(TOP_K, LEVEL_NUM, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), 4, "M2D");
    double m2d_res1 = estimator3.update(tuples);
    cout << double(tuples.size()) / m2d_res1 / 1000000 << endl;

    hierarchy::MomentEstimatorCombined estimator4(TOP_K, LEVEL_NUM, STAGE_NUM, STAGE_ROW, STAGE_COL, rd(), 8, "M2D");
    double m2d_res2 = estimator4.update(tuples);
    cout << double(tuples.size()) / m2d_res2 / 1000000 << endl;





//    cout << "pkt cnt: " << tuples.size() << tuples.back().first << tuples.back().second << endl;
}