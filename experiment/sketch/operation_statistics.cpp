#include "../../src/sketch/Aton_vHLL.hpp"
#include "../../src/sketch/Aton_vHLL_AVX.hpp"
#include "../../src/sketch/On_vHLL.hpp"
#include "../../src/sketch/On_vLLC.hpp"
#include "../../src/sketch/Ton_vHLL.hpp"
#include "../../src/sketch/vHLL.hpp"
#include "../../src/utils/ThreadPool.hpp"
#include "fmt/core.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>

using namespace std;

std::random_device rd;
spdlog::level::level_enum log_level = spdlog::level::info;

const size_t STAGE_NUM = 4;
const size_t STAGE_ROW = 8192;
const size_t STAGE_COL = 2048;

const size_t THREAD_NUM = std::thread::hardware_concurrency();

const size_t REPEAT_TIME = 4;
const size_t MAX_PKG = 10000000;
const uint32_t MASTER_SEED = rd();

stringstream global_oss;

void initOOS(string data_path) {
    global_oss.clear();
    string src_ip, dst_ip;
    ifstream ifs(data_path);
    while (ifs.good()) {
        ifs >> src_ip >> dst_ip;
        if (ifs.eof()) {
            break;
        }
        global_oss << src_ip << " " << dst_ip << std::endl;
    }
}

inline void resetOOS() {
    global_oss.clear();
    global_oss.seekg(0);
}

void task_function() {
    std::string src_ip, dst_ip;

    for (std::size_t i = 0; i < REPEAT_TIME; ++i) {
        vHLL *sketch = new vHLL(STAGE_NUM * STAGE_ROW * STAGE_COL, STAGE_COL * STAGE_NUM, rd());
//        On_vLLC *sketch = new On_vLLC(STAGE_NUM, STAGE_ROW, STAGE_COL, rd());
//        On_vHLL *sketch = new On_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL, rd());
//        Ton_vHLL *sketch = new Ton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL / 2, rd());
//        Aton_vHLL *sketch = new Aton_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL/2, rd());
//        Aton_vHLL_AVX *sketch = new Aton_vHLL_AVX(STAGE_NUM, STAGE_ROW, STAGE_COL/2, rd());

        resetOOS();

        std::size_t fcnt = 0;
        while (global_oss.good()) {
            global_oss >> src_ip >> dst_ip;
            if (global_oss.fail())
                break;
            fcnt++;
            if (fcnt > MAX_PKG) {
                break;
            }
            sketch->flowTrace(src_ip.c_str(), src_ip.length(), dst_ip.c_str(),dst_ip.length());
        }
        sketch->saveStatistic(MAX_PKG);
        delete sketch;
    }
}


int main() {
    std::string data_path = "../../data/caida/equinix-chicago.dirA.20160121-130100.UTC.anon.pcap.dat";
    initOOS(data_path);
    task_function();
    return 0;
}
