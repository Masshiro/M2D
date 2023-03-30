//
// Created by Masshiro on 2022/10/15.
//
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <random>
#include <unordered_map>

#include "../../src/sketch/Ton_vHLL.hpp"
#include "../../src/utils/CardinalityMap.hpp"
#include "../../src/utils/ThreadPool.hpp"

//const std::string INPUT_FILE_LIST = "../../data/130input_list.txt"; const int FILE_NUM = 130;
const std::string INPUT_FILE_LIST = "../../data/5input_list.txt"; const int FILE_NUM = 5;
const std::string DATA_DIR = "../../data/caida/";
const std::string OUTPUT_DIR = "../../result/hierarchy/multilayer_unbiasness/";
const int MAX_LAYER_NUM = 15;
const int LOWER_BOUND_FILTER = 10;

std::random_device rd;
spdlog::level::level_enum log_level = spdlog::level::info;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;
const uint32_t STAGE_COL = 2048;
const size_t THREAD_NUM = std::thread::hardware_concurrency();
std::mutex update_mut;
const size_t REPEAT_TIME = 10;
const uint32_t MASTER_SEED = rd();

CardinalityMap * global_cards = new CardinalityMap[MAX_LAYER_NUM];

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

void task_function(const std::string& data_path, const std::string& truth_path, int file_index, int times_cnt) {
    Ton_vHLL ** sketches = new Ton_vHLL * [MAX_LAYER_NUM];
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        sketches[i] = new Ton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL, rd());
    }
    CardinalityMap * local_cards = new CardinalityMap[MAX_LAYER_NUM];
    std::vector<std::vector<std::string> > sampled_condition(MAX_LAYER_NUM);
    std::unordered_map<std::string, double> true_card_list;

    std::ifstream infs(data_path);
    std::ifstream true_infs(truth_path);
    double r_err, est_card, true_card;
    std::string src_ip, dst_ip;
    if (!infs) {
        spdlog::error("open {} fail", data_path);
        infs.close();
        true_infs.close();
    }
    infs.clear();
    infs.seekg(0);
    true_infs.clear();
    true_infs.seekg(0);

    while (infs.good()) {
        infs >> src_ip >> dst_ip;
        if (infs.eof())
            break;
        XXHash32 hash32(MASTER_SEED);
        hash32.add(src_ip.c_str(), src_ip.length());
        uint32_t flow_id_hash = hash32.hash();
        uint32_t j_star = get_leading_ones(flow_id_hash, MAX_LAYER_NUM - 1);

        sampled_condition[j_star].emplace_back(src_ip);
        sketches[j_star]->offerFlow(src_ip.c_str(), src_ip.length(), dst_ip.c_str(),dst_ip.length());
    }


//    std::cout << "File " << file_index << " " << times_cnt << "-th times. Sketches Built!" << std::endl;


    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof())
            break;
        if (true_card < LOWER_BOUND_FILTER)
            continue;
        true_card_list[src_ip] = true_card;
    }

//    std::cout << "File " << file_index << " " << times_cnt << "-th times. Hash Table Built!" << std::endl;

    for (int i = 1; i < MAX_LAYER_NUM; ++i) {
        double cnt = 0;
        for (auto const sampled_ip : sampled_condition[i]) {
            cnt++;
            if (true_card_list.count(sampled_ip) == 0) continue;
//            std::cout << "Processing " << sampled_ip;
            est_card = sketches[i]->decodeFlow(sampled_ip.c_str(), sampled_ip.length());
            r_err = get_r_error((double)true_card_list[sampled_ip], est_card);
//            std::cout << " error: " << r_err << ' ' << cnt/double(sampled_condition[i].size())*100 << "%" << std::endl;
            local_cards[i].add((double)true_card_list[sampled_ip], r_err);
        }
    }

//    std::cout << "File " << file_index << " " << times_cnt << "-th times. Local Card Map Built." << std::endl;

//    std::lock_guard<std::mutex> lk(update_mut);
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
//        std::cout << "merging global at level " << i <<std::endl;
        global_cards[i].merge(local_cards[i]);
    }

    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        delete sketches[i];
    }
    delete sketches;
    std::cout << "File " << file_index << " " << times_cnt << "-th times. Done." << std::endl;
//    spdlog::info("task: file {} {}-th time. Done", file_index, times_cnt);
}

int main() {
    std::time_t start = time(nullptr);
    spdlog::set_level(log_level);
    std::string readline;
    int file_index = 0;
    std::ifstream ifls(INPUT_FILE_LIST);

//    while (ifls.good()) {
//        ifls >> readline;
//        if (ifls.eof())
//            break;
//        file_index ++;
//        std::string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
//        std::string true_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);
//
//        for (size_t i = 0; i < REPEAT_TIME; ++i) {
//            task_function(input_data_path, true_data_path, file_index, i);
//        }
//    }

    ThreadPool threadpool(THREAD_NUM);
    threadpool.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof())
            break;
        file_index ++;
        std::string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        std::string true_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);

        for (size_t i = 0; i < REPEAT_TIME; ++i) {
            auto taskfun = std::bind(task_function, input_data_path, true_data_path, file_index, i);
            threadpool.appendTask(taskfun);
        }
    }
    ifls.close();
    threadpool.stop();

    std::string sketchname = "ton_vhll";
    time_t tt = time(nullptr);
    tm *tmstruct = localtime(&tt);
    std::string timestr = fmt::format("{}-{}-{}", tmstruct->tm_mon + 1, tmstruct->tm_mday, tmstruct->tm_hour);
    std::string card_map_save_path = fmt::format("{}_{}s_{}r_{}c_{}.txt", sketchname, STAGE_NUM, STAGE_ROW, STAGE_COL, timestr);

    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        global_cards[i].save_to_file(fmt::format("{}_TEST_{}{}_{}", OUTPUT_DIR, "level_", i, card_map_save_path));
    }

    double cost_time = time(nullptr) - start;
    std::cout << "The experiment costs " << cost_time / 3600 << " hrs in total" << std::endl;

    return 0;
}