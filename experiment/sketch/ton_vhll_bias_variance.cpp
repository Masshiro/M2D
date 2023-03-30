//
// Created by Masshiro on 2022/9/22.
//
#include "../../src/sketch/Ton_vHLL.hpp"
#include "../../src/utils/CardinalityMap.hpp"
#include "../../src/utils/ThreadPool.hpp"
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

std::random_device rd;

spdlog::level::level_enum log_level = spdlog::level::info;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;
const uint32_t STAGE_COL = 2048;
const size_t THREAD_NUM = std::thread::hardware_concurrency();
//const size_t THREAD_NUM = 1;
const size_t REPEAT_TIME = 100;
const uint32_t MASTER_SEED = rd();

const int LOWER_BOUND_FILTER = 10;
CardinalityMap global_card;

const string INPUT_FILE_LIST = "../../data/130input_list.txt";
//const string INPUT_FILE_LIST = "../../data/5input_list.txt";
const string DATA_DIR = "../../data/caida/";
const string OUTPUT_DIR = "../../result/sketch/bias/";

void task_function(const string& data_path, const string& groundtruth_path, uint32_t label) {
    Ton_vHLL *sketch = new Ton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL / 2, rd());
    CardinalityMap tmpcard;
    spdlog::debug("{} - {}", data_path, groundtruth_path);
    ifstream infs(data_path);
    ifstream true_infs(groundtruth_path);
    int true_card = 0;
    double est_card;
    double r_err;
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
        // spdlog::debug("query : {} :> {} : {}", src_ip, est_card, true_card);
        r_err = get_r_error((double)true_card, est_card);
        tmpcard.add(true_card, r_err);
    }
    spdlog::info("task:{} end", label);
    infs.close();
    true_infs.close();
//    sketch->saveStatistic();
    delete sketch;
    global_card.merge(tmpcard);
}

int main() {
    spdlog::set_level(log_level);
    string readline;
    ThreadPool threadpool(THREAD_NUM);
    ifstream ifls(INPUT_FILE_LIST);
    uint32_t seed = 1;
    threadpool.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof())
            break;
        string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        string true_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);

        for (uint32_t i = 0; i < REPEAT_TIME; ++i) {
            auto taskfun = std::bind(task_function, input_data_path, true_data_path, ++seed);
            threadpool.appendTask(taskfun);
        }
    }
    ifls.close();
    threadpool.stop();

    string sketchname = "ton_vhll";
    time_t tt = time(nullptr);
    tm *tmstruct = localtime(&tt);

    string timestr = fmt::format("{}-{}-{}-{}", tmstruct->tm_mon + 1, tmstruct->tm_mday, tmstruct->tm_hour, tmstruct->tm_min);
    string card_map_save_path = fmt::format("{}{}_{}s_{}r_{}c_{}.txt", OUTPUT_DIR, sketchname, STAGE_NUM, STAGE_ROW, STAGE_COL, timestr);
    spdlog::info(card_map_save_path);
    global_card.save_to_file(card_map_save_path);

    return 0;
}
