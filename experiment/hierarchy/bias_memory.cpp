//
// Created by Masshiro on 2023/3/11.
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


#include "../../src/hierarchy/MomentEstimator.hpp"

using namespace std;

//===============================================================
//  Experiment Settings
//===============================================================
const string TEST_STAMP = "bias_mem-";
const uint8_t REPEAT_TIMES = 1;
const uint8_t MOMENT_COUNT = 16;

//const vector<int> moment_degree = {1, 3, 5, 7, 10, 12, 15};
const vector<double> prog_rate = {2, 4, 8};
const vector<int> level_cnt = {6, 3, 2};
const uint32_t TOP_K = 100;
const uint32_t STAGE_NUM = 4;

vector<vector<int>> sketch_settings {
    {2048, 512}, {2048, 1024}, {4096, 1024}, {4096, 2048}, {8192, 2048}
};
std::random_device rd;


//===============================================================
//  File Stream Settings
//===============================================================
const string DATA_DIR = "../../data/";

const string true_caida = "caida_ip_card.txt";
const string data_caida = "caida.txt";

vector<double> true_moments(MOMENT_COUNT);

void calculate_ground_truth() {
    ifstream in_file (DATA_DIR + true_caida);
    if (!in_file) {
        cout << "open " << true_caida << " failed." << endl;
        in_file.close();
    }
    double true_card = 0;
    string src_ip;

    while (in_file.good()) {
        in_file >> src_ip >> true_card;
        if (in_file.eof()) {
            break;
        }
        for (int i = 0; i < MOMENT_COUNT; ++i) {
            true_moments[i] += pow(true_card, i);
        }
    }
}

vector<vector<vector<vector<double>>>> AllResults(
        prog_rate.size(), vector<vector<vector<double>>> (
                1, vector<vector<double>> (
                        sketch_settings.size(), vector<double> (
                                1+MOMENT_COUNT, 0.0))) );

std::vector<std::vector<double> > moments_ground_truth(1, std::vector<double>(MOMENT_COUNT, 0.0));

double calculate_relative_error(double truth_data, double estimated_data) {
    return double((estimated_data - truth_data) / truth_data);
}

void run_one_file_data_once(string filename, int file_index) {
    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < sketch_settings.size(); ++j) {
            hierarchy::MomentEstimator<On_vHLL>* sketchPro = new hierarchy::MomentEstimator<On_vHLL>(
                    TOP_K, level_cnt[i], STAGE_NUM, sketch_settings[j][0], sketch_settings[j][1],
                    rd(), prog_rate[i]
            );

            HLL hll(sketch_settings[j][1], rd());

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
                hll.offerFlow(src_ip.c_str(), src_ip.length());
            }

//            AllResults[i][file_index][j][0] += calculate_relative_error(true_moments[0], hll.decodeFlow());

            for (int k = 0; k < MOMENT_COUNT; ++k) {
                double tmp = calculate_relative_error(
                        true_moments[k],
                        sketchPro->calculate_moment_power(hierarchy::G_sum, k));//moment_degree[k]
                AllResults[i][file_index][j][k] += tmp;
            }
            AllResults[i][file_index][j].back() += sketchPro->memory_usage_int_bits() / 8388608.0;

            delete sketchPro;
        }
    }
}


//===============================================================
// Exporting Settings
//===============================================================
const std::string OUTPUT_DIR = "../../result/hierarchy/bias_memory/";
const vector<string> moment_name = {"-1-", "-3-", "-5-", "-7-", "-10-", "-12-", "-15-"};
const vector<string> prog_rate_str = {"0.5", "0.25", "0.125"};




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

void print_results(){
    std::cout << "Read " << AllResults.size() << " files in total. " << std::endl;
    for (int i = 0; i < AllResults.size(); ++i){
        std::cout << "Progressive Probability is: " << prog_rate[i] << std::endl;
        for (int j = 0; j < AllResults[i].size(); ++j){
            std::cout << "File " << j+1 << "result: " << std::endl;
            for (int k = 0; k < AllResults[i][j].size(); ++k){
                std::cout << "\t Parameter Setting: " << sketch_settings[k][0] << ' ' << sketch_settings[k][1] << std::endl;
                for (int l = 0; l < AllResults[i][j][k].size()-1; ++l) {
                    std::cout << "\t\t R-Error of L-" << l << " moment: " << AllResults[i][j][k][l] << std::endl;
                }
                std::cout << "\t\t Memory Cost is: " << AllResults[i][j][k].back() << " MB" << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void export_results() {
    std::ofstream* outfs1 = new std::ofstream [prog_rate.size() * MOMENT_COUNT];
    std::ofstream* outfs2 = new std::ofstream [prog_rate.size() * MOMENT_COUNT];

    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < MOMENT_COUNT; ++j) {
            outfs1[i*MOMENT_COUNT+j].open(OUTPUT_DIR + TEST_STAMP + prog_rate_str[i] + moment_name[j] + "moment_bias.txt");
            outfs2[i*MOMENT_COUNT+j].open(OUTPUT_DIR + TEST_STAMP + prog_rate_str[i] + moment_name[j] + "moment_rmsre.txt");
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
                    outfs1[i*MOMENT_COUNT+l] << AllResults[i][j][k][l] << ' ' << AllResults[i][j][k].back() << std::endl;
                }
            }
        }
    }

    for (int i = 0; i < prog_rate.size(); ++i) {
        for (int j = 0; j < MOMENT_COUNT; ++j) {
            outfs1[i*MOMENT_COUNT+j].close();
            outfs2[i*MOMENT_COUNT+j].close();
        }
    }

    std::cout << "Results have been exported in folder: " << OUTPUT_DIR << std::endl;

    delete[] outfs1;
    delete[] outfs2;
}

int main(){
    calculate_ground_truth();

    run_one_file_data_once(DATA_DIR + data_caida, 0);

    average_results();

    print_results();
}