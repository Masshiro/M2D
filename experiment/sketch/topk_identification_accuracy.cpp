//
// Created by Masshiro on 2022/9/28.
//
#include "../../src/sketch/vHLL.hpp"
#include "../../src/sketch/On_vHLL.hpp"
#include "../../src/sketch/On_vLLC.hpp"
#include "../../src/sketch/Ton_vHLL.hpp"
#include "../../src/utils/CardinalityMap.hpp"
#include "../../src/utils/ThreadPool.hpp"
#include "../../src/utils/HierarchyUtils.hpp"
#include "fmt/core.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include <chrono>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;
random_device rd;

spdlog::level::level_enum log_level = spdlog::level::info;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;
const uint32_t STAGE_COL = 2048;
const size_t THREAD_NUM = std::thread::hardware_concurrency();

const size_t REPEAT_TIME = 100;
const uint32_t MASTER_SEED = rd();
const int LOWER_BOUND_FILTER = 10;
const int TOPK_MAX = 1000;
const uint8_t TOPK_LOWER_BOUND = 1;// 1
const uint8_t TOPK_HIGHER_BOUND = 100;// 6

const string INPUT_FILE_LIST = "../../data/50input_list.txt";
//const string INPUT_FILE_LIST = "../../data/5input_list.txt";
const string DATA_DIR = "../../data/caida/";
const string OUTPUT_DIR = "../../result/sketch/topk_error/";
string sketch_name;

std::mutex update_mut;
std::mutex cout_mut;

vector<unordered_map<string, int> > files_hash_index;
vector<double> accuracy(TOPK_HIGHER_BOUND-TOPK_LOWER_BOUND+1, 0);
vector<vector<pair<string, int>>> files_topk_lists;

vector<pair<string, int> > true_top_list;

class TopkList {
private:
    mutex mtx;
    unordered_map<string, double> est_top_list;
//    vector<pair<string, double> > sorted_top_list;

public:
    void add(string flow_id, double est_card) {
        lock_guard<mutex> lk(mtx);
        if (est_top_list.count(flow_id) == 0) {
            est_top_list[flow_id] = est_card;
        } else {
            est_top_list[flow_id] += est_card;
        }
    }

    void merge(unordered_map<string, double>& est_list) {
        lock_guard<mutex> lk(mtx);
        for (auto item: est_list) {
            if (est_top_list.count(item.first) == 0) {
                est_top_list[item.first] = item.second;
            } else {
                est_top_list[item.first] += item.second;
            }
        }
    }

    unordered_map<string, double> get_top_list() {
        return est_top_list;
    }

    vector<pair<string, double> > build_sorted_list() {
        vector<pair<string, double> > sorted_top_list;
        for (auto & item: est_top_list) {
            sorted_top_list.emplace_back(make_pair(item.first, item.second / REPEAT_TIME));
        }
        sort(sorted_top_list.begin(), sorted_top_list.end(), hierarchy::CompareByValue<string, double>());
        return sorted_top_list;
    }
};

TopkList global_est_top_list;

void build_truth_lists() {
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
            if (true_card >= LOWER_BOUND_FILTER) {
                true_top_list.emplace_back(make_pair(src_ip, true_card));
            } else continue;
        }
//        sort(topk_list.begin(), topk_list.end(), hierarchy::CompareByValue<string, int>());
////        int tmp = topk_list.size();
////        tmp = min(tmp, TOPK_MAX);
////        for (int i = 0; i < topk_list.size(); ++i) {
////            one_file_result[topk_list[i].first] = i;
////        }
//
//        files_hash_index.emplace_back(one_file_result);
//        files_topk_lists.emplace_back(topk_list);
    }
    sort(true_top_list.begin(), true_top_list.end(), hierarchy::CompareByValue<string, int>());
}

//void export_result(vector<pair<string, double> > topk_list, int file_index, int round_num) {
//    ofstream outf;
//    outf.open(OUTPUT_DIR + sketch_name +"-topk_list-f"+ to_string(file_index)+ "-" +to_string(round_num) + ".txt");
//
//    for (auto item: topk_list) {
//        outf << item.first << ' ' << item.second << endl;
//    }
//}

void task_function(const string& data_path, const string& groundtruth_path, int file_index, int round_count) {

    //  choose one sketch
//    vHLL *sketch = new vHLL(STAGE_NUM * STAGE_ROW * STAGE_COL, STAGE_COL * STAGE_NUM, rd()); sketch_name = "vhll";
//    On_vLLC *sketch = new On_vLLC(STAGE_NUM, STAGE_ROW, STAGE_COL, rd()); sketch_name = "on_vllc";
//    On_vHLL *sketch = new On_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL, rd()); sketch_name = "on_vhll";
    Ton_vHLL *sketch = new Ton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL * 2, rd()); sketch_name = "ton_vhll";

    spdlog::debug("{} - {}", data_path, groundtruth_path);
    ifstream infs(data_path);
    ifstream true_infs(groundtruth_path);
    int true_card = 0;
    double est_card;

    string src_ip;
    string dst_ip;
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
        // spdlog::info("{} -> {}", src_ip, dst_ip);
        if (infs.eof())
            break;
        sketch->offerFlow(src_ip.c_str(), src_ip.length(), dst_ip.c_str(),dst_ip.length());
    }


    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof())
            break;
        if (true_card < LOWER_BOUND_FILTER)
            continue;
        est_card = sketch->decodeFlow(src_ip.c_str(), src_ip.length());
        global_est_top_list.add(src_ip, est_card);
    }

    delete sketch;

    std::lock_guard<std::mutex> cout_lk(cout_mut);
    std::cout << "File " << data_path << round_count+1 << "th time. Done" << std::endl;
}

double cal_fnr_for_topK(int k) {
    vector<pair<string, double> > sorted_est_list = global_est_top_list.build_sorted_list();
    unordered_map<string, double> topk_hash_table;
    for (int i = 0; i < k; ++i) {
        topk_hash_table[sorted_est_list[i].first] = sorted_est_list[i].second;
    }

    double left_cnt = k;
    for (int i = 0; i < k; ++i) {
        string target = true_top_list[i].first;
        if (topk_hash_table.count(target) != 0) {
            left_cnt --;
        }
    }
    return left_cnt / double(k);
}

void export_results() {
    ofstream outf;
    outf.open(OUTPUT_DIR + sketch_name +".txt");

    for (int i = TOPK_LOWER_BOUND; i < TOPK_HIGHER_BOUND+1; ++i) {
//        outf << i*10 << ' ' << accuracy[i-1] << endl;
        outf << i*10 << ' ' << cal_fnr_for_topK(i*10) << endl;
    }
}

int main(){
    build_truth_lists();

    ifstream input_list;
    string one_file;
    input_list.open(INPUT_FILE_LIST);

    time_t start = time(nullptr);
    int file_index = 0;

    ThreadPool threadPool(THREAD_NUM);
    threadPool.start();
    this_thread::sleep_for(std::chrono::milliseconds(500));

    while (input_list.good()){
        input_list >> one_file;
        if (input_list.eof()){
            break;
        }
//        std::string filename = fmt::format("{}{}uniq", DATA_DIR, one_file);
        string input_data_path = fmt::format("{}{}uniq", DATA_DIR, one_file);
        string true_data_path = fmt::format("{}{}groundtruth", DATA_DIR, one_file);
        std::cout << "Processing file: " << input_data_path <<std::endl;

        for (int i = 0; i < REPEAT_TIME; ++i) {
            auto taskfun = std::bind(task_function, input_data_path, true_data_path, file_index, i);
            threadPool.appendTask(taskfun);
        }
        file_index++;
    }
    input_list.close();
    threadPool.stop();

    export_results();

    double cost_time = time(nullptr) - start;
    std::cout << "The experiment costs " << cost_time / 3600 << " hrs in total" << std::endl;
}