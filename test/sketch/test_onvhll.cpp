#include "../../src/sketch/On_vHLL.hpp"
#include "../../src/utils/CardinalityMap.hpp"
#include "../../src/utils/xxhash32.h"
#include "fmt/core.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;

std::random_device rd;

 spdlog::level::level_enum log_level = spdlog::level::info;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 4096;
const uint32_t STAGE_COL = 256;
const size_t THREAD_NUM = 1;
//const size_t THREAD_NUM = std::thread::hardware_concurrency();

const size_t REPEAT_TIME = 1;
const uint32_t MASTER_SEED = rd();

const int LOWER_BOUND_FILTER = 0;

// const string INPUT_FILE_LIST = "./data/130input_list.txt";
const string INPUT_FILE_LIST = "./data/5input_list.txt";
const string DATA_DIR = "./data/caida/";
const string OUTPUT_DIR = "./result/";

void task_function(string data_path, string groundtruth_path, uint32_t seed) {
    On_vHLL sketch(STAGE_NUM, STAGE_ROW, STAGE_COL, seed);
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

    for (size_t round = 0; round < REPEAT_TIME; ++round) {
        sketch.resetSketch();
        sketch.resetSeed(XXHash32::hash(&round, sizeof(round), seed ^ MASTER_SEED));

        infs.clear();
        infs.seekg(0);
        true_infs.clear();
        true_infs.seekg(0);
        spdlog::info("task:{} round:{}/{}", seed, round, REPEAT_TIME);

        while (infs.good()) {
            infs >> src_ip >> dst_ip;
            spdlog::info("{} -> {}", src_ip, dst_ip);
            if (infs.eof())
                break;
            sketch.offerFlow(src_ip.c_str(), src_ip.length(),
                             dst_ip.c_str(),dst_ip.length());
        }

        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof())
                break;
            if (true_card < LOWER_BOUND_FILTER)
                continue;
            est_card = sketch.decodeFlow(src_ip.c_str(), src_ip.length());
            spdlog::debug("{} :> {}", src_ip, est_card);
            r_err = get_r_error((double)true_card, est_card);
            spdlog::debug("{} -> {}", src_ip, r_err);
        }
    }
    infs.close();
    true_infs.close();
}

int main(int argc, char *argv[]) {
    string input = "data/caida/equinix-chicago.dirA.20160121-130000.UTC.anon.pcap.dat";
    string ground_truth = "data/caida/equinix-chicago.dirA.20160121-130000.UTC.anon.pcap.groundtruth";
    task_function(input, ground_truth, 45);
}
