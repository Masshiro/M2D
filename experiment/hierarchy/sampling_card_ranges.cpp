//
// Created by Masshiro on 2023/3/13.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <mutex>
#include <thread>
#include <ctime>
#include <unordered_map>

#include "../../src/utils/ThreadPool.hpp"
#include "../../src/utils/xxhash32.h"
#include "../../src/filter/MapImplFilter.hpp"
#include "fmt/core.h"

using namespace std;

//===============================================================
//  Experiment Settings
//===============================================================
const string TEST_STAMP = "card_range";
const int REPEAT_TIMES = 1;

const string DATA_DIR = "../../data/caida/";
const string OUTPUT_FOLDER = "../../result/hierarchy/sampling_card_ranges/";

//// data source
//const string INPUT_FILE_LIST = "../../data/5input_list.txt";
//const int FILE_NUM = 5;

const std::string INPUT_FILE_LIST = "../../data/50input_list.txt";
const int FILE_NUM = 50;

//const std::string INPUT_FILE_LIST = "../../data/130input_list.txt";
//const int FILE_NUM = 130;

//// sampling setting
const int level_cnt = 3;
const int filter_size = 100;
const double prog_rate = 4;


//===============================================================
// Global Variable for Storing the Results
//===============================================================
//vector<unordered_map<string, double>> quick_check_card_dict(FILE_NUM);
unordered_map<double, double> all_files_card_freq;
map<double, double> all_files_card_freq_map;
vector<unordered_map<double, double>> progressive_all_files_results(FILE_NUM);
vector<unordered_map<double, double>> uniform_all_files_results_10(FILE_NUM);
vector<unordered_map<double, double>> uniform_all_files_results_20(FILE_NUM);
vector<unordered_map<double, double>> uniform_all_files_results_30(FILE_NUM);
vector<unordered_map<double, double>> uniform_all_files_results_40(FILE_NUM);
vector<unordered_map<double, double>> uniform_all_files_results_50(FILE_NUM);
vector<vector<int>> uniform_all_sampled_cnt(FILE_NUM, vector<int>(5, 0));

map<double, double> progressive_sampling_res;
vector<map<double, double>> uniform_sampling_res(5);



//===============================================================
//  Experiment Implementation
//===============================================================

void calculate_ground_truth() {
    int file_index = 0;
    std::ifstream input_list(INPUT_FILE_LIST);
    std::string one_file;

    if (!input_list) {
        std::cout << "open " << INPUT_FILE_LIST << " failed." << std::endl;
        input_list.close();
    }
    while (input_list.good()) {
        input_list >> one_file;
        if (input_list.eof()){
            break;
        }

        //  Start to process one file
        std::string filename = fmt::format("{}{}groundtruth", DATA_DIR, one_file);
        std::ifstream true_infs(filename);
        if (!true_infs) {
            std::cout << "open " << filename << " failed." << std::endl;
            true_infs.close();
        }
        double true_card = 0;
        std::string src_ip;

        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof()) break;
            ++all_files_card_freq[true_card];
        }
        file_index ++;
    }
    for (auto it: all_files_card_freq) {
        all_files_card_freq_map.insert(it);
    }

}

int get_sample_level(uint32_t hash_val) {
    //// return an index, range in [1, level_cnt-1]
    bitset<32> bi_hash_val = hash_val;
    int lead_cnt = 0;
    for (; lead_cnt < 32; ++lead_cnt) {
        if (bi_hash_val[lead_cnt] == 1) break;
    }
    ++lead_cnt;
    if (lead_cnt % (int(prog_rate) / 2) != 0) return -1;
    int idx = lead_cnt / (prog_rate / 2);
    return std::min(idx, int(level_cnt-1));
}

void progressive_sampling(string filename, int file_index, int& seed) {
    hierarchy::MapImplFilter<string> filter(filter_size, level_cnt);

    ifstream infs(filename);
    string src_ip;
    double true_card;
    if (!infs) {
        std::cout << "Open file \' "<< filename <<"\' failed!"<<std::endl;
        infs.close();
    }

    infs.clear();
    infs.seekg(0);

    while(infs.good()){
        infs >> src_ip >> true_card;
        XXHash32 hash32(seed);
        hash32.add(src_ip.c_str(), src_ip.length());
        filter.insert_element_at_level_k(src_ip, true_card, 0);
        int j_star = get_sample_level(hash32.hash());
        if (j_star == -1) continue;
        filter.insert_element_at_level_k(src_ip, true_card, j_star);
    }

    auto one_file_res = filter.get_heaps_card_freq_info();
    for (auto it: one_file_res) {
        progressive_all_files_results[file_index][it.first] += it.second;
    }
}

void uniform_sampling(string filename, int file_index, int& seed) {
    hierarchy::MapImplFilter<string> filter(level_cnt*filter_size, 5);

    vector<int> sampled_cnt(5, 0);
    ifstream infs(filename);
    string src_ip;
    double true_card;
    if (!infs) {
        std::cout << "Open file \' "<< filename <<"\' failed!"<<std::endl;
        infs.close();
    }

    infs.clear();
    infs.seekg(0);

    while(infs.good()){
        infs >> src_ip >> true_card;
        XXHash32 hash32(seed);
        hash32.add(src_ip.c_str(), src_ip.length());
        double hash_val = hash32.hash() / ((4294967296.0));
        if (hash_val < 0.1) {
            ++sampled_cnt[0];
            filter.insert_element_at_level_k(src_ip, true_card, 0);
//            ++uniform_all_files_results_10[file_index][true_card];
        }
        else if (hash_val < 0.2) {
            ++sampled_cnt[1];
            filter.insert_element_at_level_k(src_ip, true_card, 1);
//            ++uniform_all_files_results_20[file_index][true_card];
        }
        else if (hash_val < 0.3) {
            ++sampled_cnt[2];
            filter.insert_element_at_level_k(src_ip, true_card, 2);
//            ++uniform_all_files_results_30[file_index][true_card];
        }
        else if (hash_val < 0.4) {
            ++sampled_cnt[3];
            filter.insert_element_at_level_k(src_ip, true_card, 3);
//            ++uniform_all_files_results_40[file_index][true_card];
        }
        else if (hash_val < 0.5) {
            ++sampled_cnt[4];
            filter.insert_element_at_level_k(src_ip, true_card, 4);
//            ++uniform_all_files_results_50[file_index][true_card];
        } else {
            continue;
        }
    }

    for (int i = 0; i < sampled_cnt.size(); ++i) {
        uniform_all_sampled_cnt[file_index][i] += sampled_cnt[i];
    }


    for (auto it: filter.get_heaps_card_freq_info_at_level_k(0)) {
        ++uniform_all_files_results_10[file_index][it.first];
    }
    for (auto it: filter.get_heaps_card_freq_info_at_level_k(1)) {
        ++uniform_all_files_results_20[file_index][it.first];
    }
    for (auto it: filter.get_heaps_card_freq_info_at_level_k(2)) {
        ++uniform_all_files_results_30[file_index][it.first];
    }
    for (auto it: filter.get_heaps_card_freq_info_at_level_k(3)) {
        ++uniform_all_files_results_40[file_index][it.first];
    }
    for (auto it: filter.get_heaps_card_freq_info_at_level_k(4)) {
        ++uniform_all_files_results_50[file_index][it.first];
    }
}

void merge_uniform_results() {
    vector<unordered_map<double, double>> temp_uniform_res(5);
    for (int i = 0; i < FILE_NUM; ++i) {
        for (auto it: uniform_all_files_results_10[i]) {
            temp_uniform_res[0][it.first] += it.second;
        }
        for (auto it: uniform_all_files_results_20[i]) {
            temp_uniform_res[1][it.first] += it.second;
        }
        for (auto it: uniform_all_files_results_30[i]) {
            temp_uniform_res[2][it.first] += it.second;
        }
        for (auto it: uniform_all_files_results_40[i]) {
            temp_uniform_res[3][it.first] += it.second;
        }
        for (auto it: uniform_all_files_results_50[i]) {
            temp_uniform_res[4][it.first] += it.second;
        }
    }

    for (int i = 0; i < 5; ++i) {
        for (auto it: temp_uniform_res[i]) {
            uniform_sampling_res[i].insert(it);
        }
    }
}

void merge_progressive_results() {
    unordered_map<double, double> temp_progressive_res;
    for (int i = 0; i < FILE_NUM; ++i) {
        for (auto it: progressive_all_files_results[i]) {
            temp_progressive_res[it.first] += it.second;
        }
    }
    for (auto it: temp_progressive_res) {
        progressive_sampling_res.insert(it);
//        progressive_sampling_res.insert(make_pair(it.first, it.second/REPEAT_TIMES));
    }
}

void display_progressive_results() {
    for (auto it: progressive_sampling_res) {
        cout << it.first << ' ' << it.second << endl;
    }
}

void display_uniform_results(int i) {
    for (auto it: uniform_sampling_res[i]) {
        cout << it.first << ' ' << it.second << endl;
    }
}

void display_true_results() {
    for (auto it: all_files_card_freq_map) {
        cout << it.first << ' ' << it.second << endl;
    }
}

void export_true_results() {
    ofstream outfile(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_true_range.txt");
    for (auto it: all_files_card_freq_map) {
        outfile << it.first << ' ' << it.second << endl;
    }
    outfile << endl;
    outfile.close();
}

void export_progressive_results() {
    ofstream outfile(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
    + to_string(REPEAT_TIMES) + "_times_prog-rate_" + to_string(1/prog_rate) + "_range.txt");
    for (auto it: progressive_sampling_res) {
        outfile << it.first << ' ' << it.second << endl;
    }
    outfile.close();
}

void export_uniform_results() {
    ofstream outfile1(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
                    + to_string(REPEAT_TIMES) + "_times_0.1_rate_range.txt");
    ofstream outfile2(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
                      + to_string(REPEAT_TIMES) + "_times_0.2_rate_range.txt");
    ofstream outfile3(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
                      + to_string(REPEAT_TIMES) + "_times_0.3_rate_range.txt");
    ofstream outfile4(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
                      + to_string(REPEAT_TIMES) + "_times_0.4_rate_range.txt");
    ofstream outfile5(OUTPUT_FOLDER + to_string(FILE_NUM) + "_files_"
                      + to_string(REPEAT_TIMES) + "_times_0.5_rate_range.txt");
    for (auto it: uniform_sampling_res[0]) {
        outfile1 << it.first << ' ' << it.second << endl;
    }
    for (auto it: uniform_sampling_res[1]) {
        outfile2 << it.first << ' ' << it.second << endl;
    }
    for (auto it: uniform_sampling_res[2]) {
        outfile3 << it.first << ' ' << it.second << endl;
    }
    for (auto it: uniform_sampling_res[3]) {
        outfile4 << it.first << ' ' << it.second << endl;
    }
    for (auto it: uniform_sampling_res[4]) {
        outfile5 << it.first << ' ' << it.second << endl;
    }
    outfile1.close();
    outfile2.close();
    outfile3.close();
    outfile4.close();
    outfile5.close();
}


int main() {
    calculate_ground_truth();
    export_true_results();

    vector<int> seeds;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 2147483645);
    for (int i = 0; i < REPEAT_TIMES; ++i) {
        seeds.emplace_back(distrib(gen));
    }

    std::ifstream input_list;
    std::string one_file;
    input_list.open(INPUT_FILE_LIST);

    int file_index = 0;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    while (input_list.good()){
        input_list >> one_file;
        if (input_list.eof()){
            break;
        }
        std::string filename = fmt::format("{}{}groundtruth", DATA_DIR, one_file);

        for (int i = 0; i < REPEAT_TIMES; ++i) {
            progressive_sampling(filename, file_index, seeds[i]);
            uniform_sampling(filename, file_index, seeds[i]);
        }
        file_index++;
    }
    input_list.close();

    merge_progressive_results();
    merge_uniform_results();

//    export_uniform_results();
    export_progressive_results();


//    display_progressive_results();
//    cout << endl;

//    display_true_results();
//    display_uniform_results(3);

//    calculate_ground_truth();
}