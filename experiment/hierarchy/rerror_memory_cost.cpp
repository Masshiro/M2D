//
// Created by Masshiro on 2023/2/26.
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

#include "../../src/hierarchy/UnivMonPro.hpp"
#include "../../src/hierarchy/LUSPro.hpp"
#include "../../src/utils/ThreadPool.hpp"

using std::string;
using std::vector;

//===============================================================
//  Experiment Settings
//===============================================================
const std::string TEST_STAMP = "exp1_rerror_mem-";
const uint8_t REPEAT_TIMES = 1;
const uint8_t MOMENT_COUNT = 7;

const vector<int> moment_degree = {1, 3, 5, 7, 10, 12, 15};
const vector<double> prog_rate = {2, 4, 8};
const vector<int> level_cnt = {6, 3, 2};
const uint32_t TOP_K = 100;

// Ton-vHLL parameter settings
const uint32_t STAGE_NUM = 4;
vector<vector<int>> sketch_settings{
        {2048, 512}, {2048, 1024}, {2048, 2048}, {2048, 4096},
        {4096, 512}, {4096, 1024}, {4096, 2048},
        {8192, 512}, {8192, 1024}, {8192, 2048}
};
std::random_device rd;
const uint32_t MASTER_SEED = rd();


//===============================================================
//  File Stream Settings
//===============================================================
const std::string DATA_DIR = "../../data/caida/";

const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
const int FILE_NUM = 5;

//const std::string INPUT_FILE_LIST = "../../data/50input_list.txt";
//const int FILE_NUM = 50;

//const std::string INPUT_FILE_LIST = "../../data/130input_list.txt";
//const int FILE_NUM = 130;


//===============================================================
//  Multithreading Settings
//===============================================================
std::mutex cout_mut;
const size_t THREAD_NUM = std::thread::hardware_concurrency();


//===============================================================
// Exporting Settings
//===============================================================
const std::string OUTPUT_DIR = "../../result/hierarchy/rerr_memory_cost/";
const vector<string> moment_name = {"-1-", "-3-", "-5-", "-7-", "-10-", "-12-", "-15-"};
const vector<string> prog_rate_str = {"0.5", "0.25", "0.125"};


//===============================================================
// Global Variable for Storing the Results
//===============================================================
vector<vector<vector<vector<double>>>> AllResults(
        prog_rate.size(), vector<vector<vector<double>>> (
        FILE_NUM, vector<vector<double>> (
        sketch_settings.size(), vector<double> (
        1+MOMENT_COUNT, 0.0))) );
std::vector<std::vector<double> > moments_ground_truth(FILE_NUM, std::vector<double>(MOMENT_COUNT, 0.0));


//===============================================================
//  Experiment Implementation
//===============================================================
double calculate_relative_error(double truth_data, double estimated_data) {
    return double((estimated_data - truth_data) / truth_data);
}

void calculate_ground_truth() {
    std::time_t start = time(nullptr);
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
//            moments_ground_truth[file_index][0] += true_card * std::log(true_card);
            for (int i = 0; i < moment_degree.size(); ++i) {
                moments_ground_truth[file_index][i] += std::pow(true_card, moment_degree[i]);
            }
        }
        file_index ++;
    }
    double cost_time = time(nullptr) - start;
    std::cout << "ground truth has been calculated with " << cost_time << " s." << std::endl;
}

void print_results(){
    std::cout << "Read " << AllResults.size() << " files in total. " << std::endl;
    for (int i = 0; i < AllResults.size(); ++i){
        std::cout << "Progressive Probability is: " << prog_rate[i] << std::endl;
        for (int j = 0; j < AllResults[i].size(); ++j){
            std::cout << "File " << j+1 << "result: " << std::endl;
            for (int k = 0; k < AllResults[i][j].size(); ++k){
                std::cout << "\t Parameter Setting: " << sketch_settings[k][0] << ' ' << sketch_settings[k][1] << std::endl;
                for (int l = 0; l < AllResults[i][j][k].size()-1; ++l) {
                    std::cout << "\t\t R-Error of L-" << moment_degree[l] << " moment: " << AllResults[i][j][k][l] << std::endl;
                }
                std::cout << "\t\t Memory Cost is: " << AllResults[i][j][k].back() << " MB" << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void export_results() {
    std::ofstream* outfs = new std::ofstream [prog_rate.size() * moment_degree.size()];

    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < moment_degree.size(); ++j) {
            outfs[i*MOMENT_COUNT+j].open(OUTPUT_DIR + TEST_STAMP + prog_rate_str[i] + moment_name[j] + "moment.txt");
        }
    }

    for (int i = 0; i < AllResults.size(); ++i) {
        // i: progressive rate
        for (int j = 0; j < AllResults[i].size(); ++j) {
            // j: file index
            for (int k = 0; k < AllResults[i][j].size(); ++k) {
                // k: sketch setting index
                for (int l = 0; l < AllResults[i][j][k].size()-1; ++l) {
                    // l: L-moment's relative error and memory cost
                    outfs[i*MOMENT_COUNT+l] << AllResults[i][j][k][l] << ' ' << AllResults[i][j][k].back() << std::endl;
                }
            }
        }
    }

    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < moment_degree.size(); ++j) {
            outfs[i*MOMENT_COUNT+j].close();
        }
    }

    std::cout << "Results have been exported in folder: " << OUTPUT_DIR << std::endl;

    delete[] outfs;
//    delete outfs;
}

void run_one_file_data_once(string filename, int file_index, int round_count) {
    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < sketch_settings.size(); ++j) {
            hierarchy::LUSSketch<Ton_vHLL>* sketchPro =
                    new hierarchy::LUSSketch<Ton_vHLL>(
                            TOP_K, level_cnt[i],
                            STAGE_NUM, sketch_settings[j][0], sketch_settings[j][1],
                            rd(), prog_rate[i]);

            std::ifstream infs(filename);
            std::string src_ip;
            std::string dst_ip;
            if(!infs){
                std::cout << "Open file \' "<< filename <<"\' failed!"<<std::endl;
                infs.close();
            }

            infs.clear();
            infs.seekg(0);

            while(infs.good()){
                infs >> src_ip >> dst_ip;
                sketchPro->update(src_ip, dst_ip);
            }

            for (int k = 0; k < MOMENT_COUNT; ++k) {
                AllResults[i][file_index][j][k] += calculate_relative_error(
                        moments_ground_truth[file_index][k],
                        sketchPro->calculate_moment_power(hierarchy::G_sum, moment_degree[k]));
            }
            AllResults[i][file_index][j].back() += sketchPro->memory_usage_int_bits() / 8388608.0;

            delete sketchPro;
        }
        std::lock_guard<std::mutex> cout_lk(cout_mut);
        std::cout << "File " << file_index << " all sketch parameters " << round_count+1 << "th time. Done (" << prog_rate_str[i] << ")" << std::endl;
    }
}

void average_results() {
    for (int i = 0; i < AllResults.size(); ++i) {
        for (int j = 0; j < AllResults[i].size(); ++j) {
            for (int k = 0; k < AllResults[i][j].size(); ++k) {
                for (int l = 0; l < AllResults[i][j][k].size(); ++l) {
                    AllResults[i][j][k][l] /= REPEAT_TIMES;
                }
            }
        }
    }
}

int main(){
    calculate_ground_truth();

//    std::cout << "Progressive Probability: " << AllResults.size() << std::endl;
//    std::cout << "File Count: " << AllResults[0].size() << std::endl;
//    std::cout << "Level Count: " << AllResults[0][0].size() << std::endl;
//    std::cout << "Moment Count: " << AllResults[0][0][0].size() << std::endl;
//
//    return 1;

    std::ifstream input_list;
    std::string one_file;
    input_list.open(INPUT_FILE_LIST);

    std::time_t start = time(nullptr);
    int file_index = 0;

    ThreadPool threadPool(THREAD_NUM);
    threadPool.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    while (input_list.good()){
        input_list >> one_file;
        if (input_list.eof()){
            break;
        }
        std::string filename = fmt::format("{}{}uniq", DATA_DIR, one_file);
//        std::cout << "Processing file: " << filename <<std::endl;

        for (int i = 0; i < REPEAT_TIMES; ++i) {
            auto taskfun = std::bind(run_one_file_data_once, filename, file_index, i);
            threadPool.appendTask(taskfun);
        }
        file_index++;
    }
    input_list.close();
    threadPool.stop();

    average_results();

    print_results();
//    export_results();

    double cost_time = time(nullptr) - start;
    std::cout << "The experiment costs " << cost_time / 3600 << " hrs in total" << std::endl;
}