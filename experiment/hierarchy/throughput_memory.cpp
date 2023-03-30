////
//// Created by Masshiro on 2023/2/24.
////
//#include <iostream>
//#include <vector>
//#include <string>
//#include <random>
//#include <cmath>
//#include <ctime>
//
//#include "../../src/hierarchy/UnivMonPro.hpp"
//#include "../../src/hierarchy/MomentEstimator.hpp"
//#include "../../src/hierarchy/LUSPro.hpp"
//#include "gtest/gtest.h"
//#include "fmt/core.h"
//
//using namespace std;
//
//const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
//const std::string DATA_DIR = "../../data/caida/";
//const int FILE_NUM = 5;
//
//const uint32_t TOP_K = 100;
//const uint32_t STAGE_NUM = 4;
//const uint32_t STAGE_ROW = 8192/2;//  w, 碰撞机率 8192
//const uint32_t STAGE_COL = 1024; // d, 单流精度 2048
//const uint32_t LEVELS = 7;
//const int REPEAT_TIMES = 1;
//std::random_device rd;
//
//pair<double, double> test_one_file(int file_index, string data_path, string truth_path) {
//    cout << " Processing file " << file_index << std::endl;
//    pair<double, double> result{0, 0};
//    hierarchy::MomentEstimator<On_vHLL>* sketchPro = new hierarchy::MomentEstimator<On_vHLL>(
//                    TOP_K, level_cnt[i], STAGE_NUM, sketch_settings[j][0], sketch_settings[j][1],
//                    rd(), prog_rate[i]
//                    );
////    hierarchy::LUSSketch<Ton_vHLL> sketchPro(TOP_K, LEVELS, STAGE_NUM, STAGE_ROW, STAGE_COL*2, rd());
//
//    ifstream true_infs(truth_path);
//    ifstream infs(data_path);
//
//    if (!infs) {
//        spdlog::error("open {} fail", data_path);
//        infs.close();
//        true_infs.close();
//    }
//    int true_card = 0;
//    string src_ip;
//    string dst_ip;
//
//    for (int i = 0; i < REPEAT_TIMES; ++i) {
//        infs.clear();
//        infs.seekg(0);
//        int pkt_cnt = 0;
//
//        clock_t start = std::clock();
//        while (infs.good()) {
//            pkt_cnt++;
//            infs >> src_ip >> dst_ip;
//            if (infs.eof()) {
//                break;
//            }
//            sketchPro.update(src_ip, dst_ip);
//        }
//        double time_cost = (double)(clock()-start) / CLOCKS_PER_SEC;
//        result.first += pkt_cnt / time_cost / 1000000.0; // convert to Mpps
//        result.second += sketchPro.memory_usage_int_bits();
//
//        cout << "packet count: " << pkt_cnt << ", time cost: " << time_cost << " s" << endl;
//    }
//    result.first /= REPEAT_TIMES;
//    result.second /= REPEAT_TIMES;
//
//    return result;
//}
//
//int main() {
//    int file_idx = 0;
//    string readline;
//    vector<pair<double, double>> results;
//
//    std::ifstream ifls(INPUT_FILE_LIST);
//    while (ifls.good()) {
//        ifls >> readline;
//        if (ifls.eof()) {
//            break;
//        }
//        std::string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
//        std::string truth_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);
//        results.emplace_back(test_one_file(++file_idx, input_data_path, truth_data_path));
//    }
//
//    cout << endl;
//    for (auto one_result: results) {
//        int name_index = 1;
//        cout << "Result of file " << name_index << " is:" << endl;
//        cout << "\t throughput: " << one_result.first << " Mpps" << endl;
//        cout << "\t memory cost: " << one_result.second << " bits" << endl;
//    }
//    cout << endl;
//}