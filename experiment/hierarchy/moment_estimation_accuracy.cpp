//
// Created by Masshiro on 2022/9/21.
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

//===============================================================
//  Experiment Settings
//===============================================================
const std::string TEST_STAMP = "lus-hash-levels-";
const uint8_t MOMENT_COUNT = 7;
//const std::vector<std::string> moment_name = {"-e-", "-1-", "-2-", "-3-", "-4-", "-5-", "-6-"};
const std::vector<std::string> moment_name = {"-10-", "-12-", "-14-", "-16-", "-18-", "-20-", "-22-"};
const std::vector<int> moment_degree = {10, 12, 14, 16, 18, 20, 22};
const uint8_t REPEAT_TIMES = 100;
const uint8_t LEVEL_LOWER_BOUND = 1;// 1
const uint8_t LEVEL_HIGHER_BOUND = 6;// 6
const uint32_t TOP_K = 100;
std::random_device rd;
const uint32_t MASTER_SEED = rd();

//===============================================================
//  File Stream Settings
//===============================================================
const std::string DATA_DIR = "../../data/caida/";

//const std::string INPUT_FILE_LIST = "../../data/5input_list.txt";
//const int FILE_NUM = 5;
const std::string INPUT_FILE_LIST = "../../data/130input_list.txt";
const int FILE_NUM = 130;

//================================================================
//const std::string DATA_DIR = "../../data/";
//const std::string INPUT_FILE_LIST = "../../data/3input_list.txt";

//const std::string TEST_PATH = "../../data/test.uniq";
//const std::string TEST_PATH = "../../data/test_20.uniq";

//===============================================================
//  OnvHLL Settings
//===============================================================
const uint32_t STAGE_NUM = 4;
const uint32_t STAGE_ROW = 8192;//1024
const uint32_t STAGE_COL = 1024;//512 //65536

//===============================================================
//  Multithreading Settings
//===============================================================
std::mutex cout_mut;
const size_t THREAD_NUM = std::thread::hardware_concurrency();

//===============================================================
// Exporting Settings
//===============================================================
const std::string OUTPUT_DIR = "../../result/hierarchy/moment_estimation_accuracy/";

//===============================================================
// Global Variable for Storing the Results
//===============================================================
//std::vector<std::vector<std::vector<double> > > AllFilesResults(FILE_NUM);
std::vector<std::vector<std::vector<double> > > AllFilesResults(FILE_NUM, std::vector<std::vector<double>>(LEVEL_HIGHER_BOUND-LEVEL_LOWER_BOUND+1, std::vector<double>(MOMENT_COUNT, 0.0)));
std::vector<std::vector<double> > moments_ground_truth(FILE_NUM, std::vector<double>(MOMENT_COUNT, 0.0));
//// Structure of 'std::vector<std::vector<std::vector<double> > > AllFilesResults'
/*
 *  vector<vector<double>> OneFileResult
 *       /                     \
 *      /                       \
 *      +-----------------------+      +-----------------------+
 *      |         File 1        |      |         File n        |
 *      +-----------------------+      +-----------------------+ \
 *      |        level 1        |      |        level 1        |  \  vector<double>
 *      |   0-m; 1-m; 2-m; e-m; |      |   0-m; 1-m; 2-m; e-m; |  /   tmp_res_vec
 *      +-----------------------+      +-----------------------+ /
 *      ========================= ...  =========================
 *      +-----------------------+      +-----------------------+
 *      |        level x        |      |        level x        |
 *      |   0-m; 1-m; 2-m; e-m; |      |   0-m; 1-m; 2-m; e-m; |
 *      +-----------------------+      +-----------------------+
 */


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

        //  normal moment calculation
//        while (true_infs.good()) {
//            true_infs >> src_ip >> true_card;
//            if (true_infs.eof()) break;
//            moments_ground_truth[file_index][0] += true_card * std::log(true_card);
//            for (int i = 1; i < MOMENT_COUNT; ++i) {
//                moments_ground_truth[file_index][i] += std::pow(true_card, i);
//            }
//        }

        //  higher moment calculation
        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof()) break;
//            moments_ground_truth[file_index][0] += true_card * std::log(true_card);
            for (int i = 0; i < MOMENT_COUNT; ++i) {
                moments_ground_truth[file_index][i] += std::pow(true_card, moment_degree[i]);
            }
        }
        file_index ++;
    }
    double cost_time = time(nullptr) - start;
    std::cout << "ground truth has been calculated with " << cost_time << " s." << std::endl;
}

void print_results(){
    std::cout << "Read " << AllFilesResults.size() << " files in total. " << std::endl;
    for (int i = 0; i < AllFilesResults.size(); ++i){
        std::cout << "File " << i+1 << std::endl;
        for (int j = 0; j < AllFilesResults[i].size(); ++j){
            std::cout << " levels: " << (j+LEVEL_LOWER_BOUND)*2 << std::endl;
            for (int k = 0; k < AllFilesResults[i][j].size(); ++k){
                std::cout << moment_degree[k] << "-th-moment result: " << AllFilesResults[i][j][k] << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void export_results(){
    std::ofstream* outfs = new std::ofstream [MOMENT_COUNT];
    std::string sketch_info_str = std::to_string(STAGE_NUM) + "s_" + std::to_string(STAGE_ROW) + "r_" + std::to_string(STAGE_COL) + "c-top-" + std::to_string(TOP_K);
    for (int i = 0; i < MOMENT_COUNT; ++i) {
        outfs[i].open(OUTPUT_DIR + TEST_STAMP + moment_name[i] + "moment.txt");
    }

    for (int i = 0; i < AllFilesResults.size(); ++i){
        for (int j = 0; j < AllFilesResults[i].size(); ++j){
            for (int k = 0; k < MOMENT_COUNT; ++k) {
                outfs[k] << AllFilesResults[i][j][k] << std::endl;
            }
        }
    }

    for (int i = 0; i < MOMENT_COUNT; ++i) {
        outfs[i].close();
    }

    delete[] outfs;
}

void run_one_file_data_once(std::string filename, int file_index, int round_count){
    for (int i = LEVEL_LOWER_BOUND; i < LEVEL_HIGHER_BOUND + 1; ++i) {
//        univmon::UnivMonSketchPro* sketchPro =
//            new univmon::UnivMonSketchPro(TOP_K, i*2,STAGE_NUM, STAGE_ROW, STAGE_COL, rd());

//        hierarchy::LUSSketch<On_vHLL>* sketchPro =
//                new hierarchy::LUSSketch<On_vHLL>(TOP_K, i*2-1,STAGE_NUM, STAGE_ROW, STAGE_COL, rd());

        hierarchy::LUSSketch<Ton_vHLL>* sketchPro =
                new hierarchy::LUSSketch<Ton_vHLL>(TOP_K, i*2-1,STAGE_NUM, STAGE_ROW, STAGE_COL*2, rd());

//        univmon::UnivMonSketchLite* sketchPro =
//            new univmon::UnivMonSketchLite(TOP_K, i*2-1,STAGE_NUM, STAGE_ROW, STAGE_COL, rd());

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
//            sketchPro->update(src_ip.c_str(), src_ip.length(), dst_ip.c_str(), dst_ip.length());
//            sketchPro->update_lus(src_ip.c_str(), src_ip.length(), dst_ip.c_str(), dst_ip.length());
            sketchPro->update(src_ip, dst_ip);
        }

        //  normal moment calculation
//        AllFilesResults[file_index][i-LEVEL_LOWER_BOUND][0] += calculate_relative_error(moments_ground_truth[file_index][0], sketchPro->calculate_moment_entropy(hierarchy::G_entropy));
//        for (int j = 1; j < MOMENT_COUNT; ++j) {
//            AllFilesResults[file_index][i-LEVEL_LOWER_BOUND][j] += calculate_relative_error(moments_ground_truth[file_index][j], sketchPro->calculate_moment_power(hierarchy::G_sum, j));
//        }

        //  higher moment calculation
        for (int j = 0; j < MOMENT_COUNT; ++j) {
            AllFilesResults[file_index][i-LEVEL_LOWER_BOUND][j] += calculate_relative_error(moments_ground_truth[file_index][j], sketchPro->calculate_moment_power(hierarchy::G_sum, moment_degree[j]));
        }

        delete sketchPro;
    }
    std::lock_guard<std::mutex> cout_lk(cout_mut);
    std::cout << "File " << filename << " all levels " << round_count+1 << "th time. Done" << std::endl;
}

void average_results(){
    for (int i = 0; i < AllFilesResults.size(); ++i) {
        for (int j = 0; j < AllFilesResults[i].size(); ++j) {
            for (int k = 0; k < AllFilesResults[i][j].size(); ++k) {
                AllFilesResults[i][j][k] /= REPEAT_TIMES;
            }
        }
    }
}

int main(){
    calculate_ground_truth();

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
        std::cout << "Processing file: " << filename <<std::endl;

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
    export_results();

    double cost_time = time(nullptr) - start;
    std::cout << "The experiment costs " << cost_time / 3600 << " hrs in total" << std::endl;
}
