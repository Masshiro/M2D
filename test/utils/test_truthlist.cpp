//
// Created by Masshiro on 2022/9/28.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "../../src/utils/TruthList.hpp"

const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
const std::string DATA_DIR = "../../data/caida/";


void test_one_file(int file_index, const std::string& data_path, const std::string& truth_path) {
    int error_count = 0;
    bool has_fault = false;
    std::cout << "\nTesting for file " << file_index << std::endl;

    TruthList<std::string> truthList;
    std::ifstream infs(data_path);
    std::ifstream true_infs(truth_path);

    int true_card = 0;
    std::string src_ip;
    std::string dst_ip;
    if (!infs) {
        spdlog::error("open {} fail", data_path);
        infs.close();
        true_infs.close();
    }

    infs.clear();
    infs.seekg(0);
    true_infs.clear();
    true_infs.seekg(0);

    while(infs.good()) {
        infs >> src_ip >> dst_ip;
        if (infs.eof()) {
            break;
        }
        truthList.insert_element_at_level_k(src_ip, dst_ip);
    }

    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof())
            break;
        std::pair<bool, int> check_result = truthList.check_one_flow_correctness_at_level_k(src_ip, true_card, 0);
        if (!check_result.first) {
            error_count ++;
            std::cout << "ERROR:\t truth:\t" << true_card << ",\t get:\t" << check_result.second << "\t source IP:\t" << src_ip << std::endl;
            has_fault = true;
        }
    }

    if (!has_fault) {
        std::cout << "\nThere is no fault in file " << file_index << std::endl;
    } else{
        std::cout << error_count <<" errors,\t" << truthList.get_hashtable_size_at_level_k(0) << " flows" << std::endl;
    }

    truthList.print_truth_list_top_n_at_level_k(300);
}

TEST(TruthList, basic) {
    int file_index = 0;
    std::string readline;

    std::ifstream ifls(INPUT_FILE_LIST);
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof()) {
            break;
        }
        std::string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        std::string truth_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);
        test_one_file(++file_index, input_data_path, truth_data_path);
    }
}