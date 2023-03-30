#include "../../src/sketch/On_vHLL.hpp"
#include "../../src/utils/CardinalityMap.hpp"
#include "../../src/utils/ThreadPool.hpp"
#include "fmt/core.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"
#include <chrono>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;

std::random_device rd;

const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;
const uint32_t STAGE_COL = 65536;

const string INPUT_FILE_LIST = "../../data/test_global_histogram_input_list.txt";
const string DATA_DIR = "../../data/caida/";

TEST(GlobalHistogram, basic) {
    string readline;
    ifstream ifls(INPUT_FILE_LIST);
    while (ifls.good()) {
        ifls >> readline;
        if (ifls.eof())
            break;
        string input_data_path = fmt::format("{}{}uniq", DATA_DIR, readline);
        string true_data_path = fmt::format("{}{}groundtruth", DATA_DIR, readline);

        On_vHLL *sketch = new On_vHLL(STAGE_NUM, STAGE_ROW, STAGE_COL, rd());
        spdlog::info("{}", readline);
        ifstream infs(input_data_path);
        ifstream true_infs(true_data_path);
        string src_ip;
        string dst_ip;
        if (!infs) {
            spdlog::error("open {} fail", input_data_path);
            infs.close();
            true_infs.close();
        }
        infs.clear();
        infs.seekg(0);
        true_infs.clear();
        true_infs.seekg(0);

        int cnt = 0;
        while (infs.good()) {
            infs >> src_ip >> dst_ip;
            cnt++;
            if (infs.eof())
                break;
            sketch->offerFlow(src_ip.c_str(), src_ip.length(), dst_ip.c_str(),dst_ip.length());
        }

        sketch->getHistogram();
        spdlog::info("true card:{}", cnt);
        infs.close();
        delete sketch;
    }
    ifls.close();
}
