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
std::mutex cout_mut;
const size_t REPEAT_TIME = 10;
const uint32_t MASTER_SEED = rd();

std::vector<std::vector<std::pair<std::string, std::string> > > uniq_files(FILE_NUM);
//std::vector<std::vector<std::pair<std::string, double> > > truth_files(FILE_NUM);
std::vector<std::unordered_map<std::string, double> > truth_files(FILE_NUM);

CardinalityMap * global_cards = new CardinalityMap[MAX_LAYER_NUM];

void load_data(){
    std::ifstream input_list(INPUT_FILE_LIST);
    std::string one_file;
    if (!input_list) {
        std::cout << "open " << INPUT_FILE_LIST << " failed!" << std::endl;
        input_list.close();
    }
    int file_index = 0;
    while (input_list.good()) {
        input_list >> one_file;
        if (input_list.eof()) {
            break;
        }

        //  start to read one file
        std::string filename_uniq = DATA_DIR + one_file + "uniq";
        std::string filename_truth = DATA_DIR + one_file + "groundtruth";

        double true_card;
        std::string src_ip;
        std::string dst_ip;

        //  load contents of src-dst ip pairs using .uniq files
        std::ifstream uniq_infs(filename_uniq);
        if (!uniq_infs) {
            std::cout << "open " << filename_uniq << " failed!" << std::endl;
            uniq_infs.close();
        }
        uniq_infs.clear();
        uniq_infs.seekg(0);
        while (uniq_infs.good()) {
            uniq_infs >> src_ip >> dst_ip;
            if (uniq_infs.eof()) break;
            uniq_files[file_index].emplace_back(std::make_pair(src_ip, dst_ip));
        }

        //  load contents of card-freq distribution using .groundtruth files
        std::ifstream true_infs(filename_truth);
        if (!true_infs) {
            std::cout << "open " << filename_truth << " failed." << std::endl;
            true_infs.close();
        }
        true_infs.clear();
        true_infs.seekg(0);
        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof()) break;
            truth_files[file_index][src_ip] = true_card;
//            truth_files[file_index].emplace_back(std::make_pair(src_ip, true_card));
        }
        file_index++;
    }
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

void task_function(int file_index, int time_count) {
    Ton_vHLL ** sketches = new Ton_vHLL * [MAX_LAYER_NUM];
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        sketches[i] = new Ton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL, rd());
    }
    CardinalityMap * local_cards = new CardinalityMap[MAX_LAYER_NUM];
    double r_err;
    double est_card;
    double true_card;

    //  update the sketch for each layer
    std::vector<std::vector<std::string> > sampled_condition(MAX_LAYER_NUM);
    for (auto const flow: uniq_files[file_index]) {
        XXHash32 hash32(MASTER_SEED);
        hash32.add(flow.first.c_str(), flow.first.length());
        uint32_t flow_id_hash = hash32.hash();
        uint32_t j_star = get_leading_ones(flow_id_hash, MAX_LAYER_NUM - 1);
        sampled_condition[j_star].emplace_back(flow.first);
        sketches[j_star]->offerFlow(flow.first.c_str(), flow.first.length(), flow.second.c_str(), flow.second.length());
    }

    //  update the cardinality map structures
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        double cnt = 0;
        double n_fast = sketches[i]->getNhatFast();
        double n_slow = sketches[i]->getNhatSlow();
        for (auto const sampled_ip: sampled_condition[i]) {
            cnt ++;
            true_card = truth_files[file_index][sampled_ip];
            if (true_card < LOWER_BOUND_FILTER) continue;
            est_card = sketches[i]->decodeFlow(sampled_ip.c_str(), sampled_ip.length());
            est_card += (1/(STAGE_ROW-1)) * (n_fast - n_slow);
            r_err = get_r_error((double)true_card, est_card);
            local_cards[i].add(true_card, r_err);

            std::lock_guard<std::mutex> lk(cout_mut);
            std::cout << "file " << file_index << " " << time_count << "-th time. layer " << i << ' ' << cnt/double(sampled_condition[i].size())*100 << "%" << std::endl;
        }
    }

    //  merge local cardinality maps to global ones layer-by-layer
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        global_cards[i].merge(local_cards[i]);
    }

    //  delete the sketches
    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        delete sketches[i];
    }
    delete[] sketches;

    std::lock_guard<std::mutex> lk(cout_mut);
    std::cout << "File " << file_index << " " << time_count << "-th times. Done." << std::endl;
}

int main() {
    std::time_t start = time(nullptr);
    load_data();
    spdlog::set_level(log_level);
    ThreadPool threadpool(THREAD_NUM);
    threadpool.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (int i = 0; i < FILE_NUM; ++i) {
        for (size_t j = 0; j < REPEAT_TIME; ++j) {
            auto taskfun = std::bind(task_function, i, j);
            threadpool.appendTask(taskfun);
        }
    }

    threadpool.stop();

    std::string sketchname = "ton_vhll";
    time_t tt = time(nullptr);
    tm *tmstruct = localtime(&tt);

    std::string timestr = fmt::format("{}-{}-{}", tmstruct->tm_mon + 1, tmstruct->tm_mday, tmstruct->tm_hour);
    std::string card_map_save_path = fmt::format("{}_{}s_{}r_{}c_{}.txt", sketchname, STAGE_NUM, STAGE_ROW, STAGE_COL, timestr);

    for (int i = 0; i < MAX_LAYER_NUM; ++i) {
        global_cards[i].save_to_file(fmt::format("{}{}_{}{}", OUTPUT_DIR, "level_", i, card_map_save_path));
    }
    double cost_time = time(nullptr) - start;
    std::cout << "The experiment costs " << cost_time / 3600 << " hrs in total" << std::endl;
}