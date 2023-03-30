#include "../src/sketch/HLL.hpp"
#include "../src/sketch/On_vHLL.hpp"
#include "../src/sketch/Ton_vHLL.hpp"
#include "../src/utils/CardinalityMap.hpp"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"
#include <ctime>
#include <random>
#include <vector>

using namespace std;

random_device rd;

const size_t REPEAT_TIME = 100;
const int LOWER_BOUND_FILTER = 100;
CardinalityMap global_card;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 1024;
const uint32_t STAGE_COL = 128;

// const string INPUT_FILE_LIST = "/home/caiyuexiao/on-vhll-tailcut/data/130input_list.txt";
const string INPUT_FILE_LIST = "/home/caiyuexiao/on-vhll-tailcut/data/5input_list.txt";
const string DATA_DIR = "/home/caiyuexiao/on-vhll-tailcut/data/caida/";
const string OUTPUT_DIR = "/home/caiyuexiao/on-vhll-tailcut/result/";
const string LOG_DIR = "/home/caiyuexiao/on-vhll-tailcut/log/";

void task_function(const string& data_path, const string& groundtruth_path, uint32_t label) {
    Ton_vHLL *sketch = new Ton_vHLL(STAGE_NUM, STAGE_ROW / 2, STAGE_COL, rd());
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
//         spdlog::info("{} -> {}", src_ip, dst_ip);
        if (infs.eof())
            break;
        sketch->offerFlow(src_ip.c_str(), src_ip.length(),
                          dst_ip.c_str(),dst_ip.length());
    }

    while (true_infs.good()) {
        true_infs >> src_ip >> true_card;
        if (true_infs.eof())
            break;
        if (true_card < LOWER_BOUND_FILTER)
            continue;
        est_card = sketch->decodeFlow(src_ip.c_str(), src_ip.length());
//        spdlog::info("query : {} :> {} : {}", src_ip, est_card, true_card);
        r_err = get_r_error((double)true_card, est_card);
        tmpcard.add(true_card, r_err);
    }
    infs.close();
    true_infs.close();
    delete sketch;
    global_card.merge(tmpcard);
    global_card.show();
}

TEST(OVHT4bit, multiflow) {
    ifstream ifls(INPUT_FILE_LIST);
    string readline;
    uint32_t seed = 1;

    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof())
            break;
        string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        string true_data_path =
            fmt::format("{}{}groundtruth", DATA_DIR, readline);
        spdlog::debug(input_data_path);

        for (size_t i = 0; i < REPEAT_TIME; ++i) {
            task_function(input_data_path, true_data_path, ++seed);
        }
    }
    ifls.close();
}