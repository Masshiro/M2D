//
// Created by Masshiro on 2023/3/3.
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

#include "../../src/hierarchy/LUSPro.hpp"
#include "../../src/utils/ThreadPool.hpp"
#include "fmt/core.h"

using std::string;
using std::vector;

//===============================================================
//  Experiment Settings
//===============================================================
const std::string TEST_STAMP = "level_mem-";
const uint8_t REPEAT_TIMES = 100;

const vector<double> prog_rate = {2, 4, 8};
const vector<string> prog_rate_str = {"0.5", "0.25", "0.125"};
const vector<int> level_cnt = {2, 3, 4, 5, 6, 7};
const uint32_t TOP_K = 100;

// Ton-vHLL parameter settings
const uint32_t STAGE_NUM = 4;
vector<vector<int>> sketch_settings{
        {2048, 512}, {2048, 1024}, {4096, 1024}, {4096, 2048}, {8192, 2048}
};
std::random_device rd;
const uint32_t MASTER_SEED = rd();


//===============================================================
//  File Stream Settings
//===============================================================
const std::string DATA_DIR = "../../data/caida/";
const std::string INPUT_FILE_LIST = "../../data/50input_list.txt";
const std::string INPUT_DATA_FILE = "../../data/generated.txt";
const int FILE_NUM = 50;
const std::string OUTPUT_DIR = "../../result/hierarchy/level_memory_cost/";


//===============================================================
//  Function to Calculate the Number of Packets
//  Sampled into Hash Table
//===============================================================
static inline uint32_t get_leading_zeros(uint32_t hash_value) {
    uint32_t mask = 1;
    uint32_t cnt = 1;
    for ( uint32_t i = 0; i < sizeof(hash_value) * 8; i++) {
        if ((mask & hash_value) != 0){
            break;
        }
        mask <<= 1;
        cnt++;
    }
    if (cnt >= 32) {
        cnt = 31;
    }
    return cnt;
}

int get_sample_level(uint32_t hash_val, double prog_rate, int level) {
    int lead_cnt = get_leading_zeros(hash_val);
    if (lead_cnt % (int(prog_rate) / 2) != 0) return -1;
    int idx = lead_cnt / (prog_rate / 2);
    return std::min(idx, int(level));
}

double calculate_hashtable_memory(double rate, int level) {
    // using CAIDA 50 files
//    int hash_table_hits = 0;
//    std::ifstream input_list(INPUT_FILE_LIST);
//    std::string one_file;
//
//    if (!input_list) {
//        std::cout << "open " << INPUT_FILE_LIST << " failed." << std::endl;
//        input_list.close();
//    }
//    while (input_list.good()) {
//        input_list >> one_file;
//        if (input_list.eof()){
//            break;
//        }
//        //  Start to process one file
//        std::string filename = fmt::format("{}{}uniq", DATA_DIR, one_file);
//        std::ifstream true_infs(filename);
//        if (!true_infs) {
//            std::cout << "open " << filename << " failed." << std::endl;
//            true_infs.close();
//        }
//
//        std::string src_ip;
//        std::string dst_ip;
//
//        while (true_infs.good()) {
//            true_infs >> src_ip >> dst_ip;
//            if (true_infs.eof()) break;
//            XXHash32 hash32(MASTER_SEED);
//            hash32.add(src_ip.c_str(), src_ip.length());
//            if (get_sample_level(hash32.hash(), rate, level) == level) {
//                ++hash_table_hits;
//            }
//        }
//    }
//    hash_table_hits /= 50.0;
//    return double(sizeof(int) * 2 * hash_table_hits * CHAR_BIT) / 8388608.0;

    // using generated data
    int hash_table_hits = 0;
    std::ifstream infile(INPUT_DATA_FILE);
    if (!infile) {
        std::cout << "open " << INPUT_DATA_FILE << " failed." << std::endl;
        infile.close();
    }
    std::string src_ip, dst_ip;
    while (infile.good()) {
        infile >> src_ip >> dst_ip;

        if (infile.eof()) {
            break;
        }

        XXHash32 hash32(MASTER_SEED);
        hash32.add(src_ip.c_str(), src_ip.length());
        if (get_sample_level(hash32.hash(), rate, level) == level) {
            ++hash_table_hits;
        }
    }

    return double(sizeof(int) * 2 * hash_table_hits * CHAR_BIT) / 8388608.0;
}

double calculate_sketch_memory(double rate, int level, vector<int> params) {
    double bits_used = 0;
    for (int i = 0; i < level; ++i) {
        bits_used += (params[0] / pow(rate, i)) * STAGE_NUM * params[1] * 4;
    }
    bits_used += (sizeof(int) + sizeof(double)) * TOP_K * CHAR_BIT;

    return bits_used / 8388608.0;
}

double calculate_hashtable_memory_theo(double rate, int level) {
    // using CAIDA 50 files
//    int pkt_cnt = 0;
//    std::ifstream input_list(INPUT_FILE_LIST);
//    std::string one_file;
//
//    if (!input_list) {
//        std::cout << "open " << INPUT_FILE_LIST << " failed." << std::endl;
//        input_list.close();
//    }
//    while (input_list.good()) {
//        input_list >> one_file;
//        if (input_list.eof()){
//            break;
//        }
//        //  Start to process one file
//        std::string filename = fmt::format("{}{}uniq", DATA_DIR, one_file);
//        std::ifstream true_infs(filename);
//        if (!true_infs) {
//            std::cout << "open " << filename << " failed." << std::endl;
//            true_infs.close();
//        }
//
//        std::string src_ip;
//        std::string dst_ip;
//
//        while (true_infs.good()) {
//            true_infs >> src_ip >> dst_ip;
//            ++pkt_cnt;
//        }
//    }
//    pkt_cnt /= 50.0;
//    return double(sizeof(int) * 2 * (pkt_cnt / pow(rate, level)) * CHAR_BIT) / 8388608.0;

    // using generated data
    int pkt_cnt = 0;
    std::ifstream infile(INPUT_DATA_FILE);
    if (!infile) {
        std::cout << "open " << INPUT_DATA_FILE << " failed." << std::endl;
        infile.close();
    }
    std::string src_ip, dst_ip;
    while (infile.good()) {
        infile >> src_ip >> dst_ip;

        if (infile.eof()) {
            break;
        }

        ++pkt_cnt;
    }

    return double(sizeof(int) * 2 * (pkt_cnt / pow(rate, level)) * CHAR_BIT) / 8388608.0;
}


int main() {
    vector<vector<vector<double>>> AllResults(
            prog_rate.size(), vector<vector<double>>(
            level_cnt.size(), vector<double>(
            sketch_settings.size(), 0.0 )));

    vector<vector<vector<double>>> TheoryResults(
            prog_rate.size(), vector<vector<double>>(
            level_cnt.size(), vector<double>(
            sketch_settings.size(), 0.0 )));

    std::ofstream* outfs = new std::ofstream [prog_rate.size()];
    for (int i = 0; i < prog_rate.size(); ++i) {
        outfs[i].open(OUTPUT_DIR + TEST_STAMP + prog_rate_str[i] + "-generated-theo.txt");
    }
//// Export experiment memory cost
//    for (int i = 0; i < prog_rate.size(); ++i) {
//        for (int j = 0; j < level_cnt.size(); ++j) {
//            double hash_table_mem = 0;
//            for (int n = 0; n < REPEAT_TIMES; ++n) {
//                hash_table_mem += calculate_hashtable_memory(prog_rate[i], level_cnt[j]);
//                std::cout << "Progressive Sampling Prob: " << 1/prog_rate[i] << ", Level: " << level_cnt[j] << ", " << n+1 << " th time. Done." << std::endl;
//            }
//            hash_table_mem /= REPEAT_TIMES;
//
//            for (int k = 0; k < sketch_settings.size(); ++k) {
//                AllResults[i][j][k] = hash_table_mem + calculate_sketch_memory(prog_rate[i], level_cnt[j], sketch_settings[k]);
//            }
//        }
//    }
//
//    for (int i = 0; i < AllResults.size(); ++i) {
//        std::cout << "Progressive rate: " << 1/prog_rate[i] << std::endl;
//        for (int j = 0; j < AllResults[i].size(); ++j) {
//            std::cout << "\t Level = " << level_cnt[j] << '\t';
//            for (int k = 0; k < AllResults[i][j].size(); ++k) {
//                std::cout << AllResults[i][j][k] << ' ';
//                outfs[i] << AllResults[i][j][k] << ' ';
//            }
//            std::cout << std::endl;
//            outfs[i] << std::endl;
//        }
//    }

//// Export theoretical memory cost
    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < level_cnt.size(); ++j) {
            double hash_table_mem = calculate_hashtable_memory_theo(prog_rate[i], level_cnt[j]);
            for (int k = 0; k < sketch_settings.size(); ++k) {
                TheoryResults[i][j][k] = hash_table_mem + calculate_sketch_memory(prog_rate[i], level_cnt[j], sketch_settings[k]);
            }
        }
    }

    for (int i = 0; i < TheoryResults.size(); ++i) {
        std::cout << "Progressive rate: " << 1/prog_rate[i] << std::endl;
        for (int j = 0; j < TheoryResults[i].size(); ++j) {
            std::cout << "\t Level = " << level_cnt[j] << '\t';
            for (int k = 0; k < TheoryResults[i][j].size(); ++k) {
                std::cout << TheoryResults[i][j][k] << ' ';
                outfs[i] << TheoryResults[i][j][k] << ' ';
            }
            std::cout << std::endl;
            outfs[i] << std::endl;
        }
    }

}