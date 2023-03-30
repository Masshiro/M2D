#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <unordered_map>
#include <algorithm>

using namespace std;

const string INPUT_FILE_LIST = "../../data/130input_list.txt";
const string DATA_DIR = "../../data/caida/";

template<class KEY, class VAL>
struct CompareByKey{
    bool operator() (const std::pair<KEY, VAL>& p1, const std::pair<KEY, VAL>& p2){
        return p1.first < p2.first;
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    //============================================================
    //  Correct the spells of the name of ground truth files
    //============================================================
//    ifstream ifls(INPUT_FILE_LIST);
//    string readline;
//    int file_cnt = 0;
//    while (ifls.good()) {
//        ifls >> readline;
//        if (ifls.eof()) {
//            break;
//        }
//        string old_file_name = DATA_DIR + readline + "grouptruth";
//        string new_file_name = DATA_DIR + readline + "groundtruth";
//        rename(old_file_name.c_str(), new_file_name.c_str());
//        cout << "FILE " << ++file_cnt << ":\t renamed." << endl;
//    }

    //============================================================
    //  Calculate the moments of all 130 files
    //============================================================
//    vector<double> moments(7, 0);
//    ifstream input_list(INPUT_FILE_LIST);
//    double pkt_count = 0;
//    string one_file;
//    if (!input_list) {
//        cout << "open " << INPUT_FILE_LIST << " failed!" <<endl;
//        input_list.close();
//    }
//    while (input_list.good()) {
//        input_list >> one_file;
//        if (input_list.eof()) {
//            break;
//        }
//
//        string filename = DATA_DIR + one_file + "groundtruth";
//        string filename_cnt = DATA_DIR + one_file + "uniq";
//
//        double true_card = 0;
//        string src_ip;
//        string dst_ip;
//
//        //  use .uniq file to count total packet number
//        ifstream cnt_infs(filename_cnt);
//        if (!cnt_infs) {
//            std::cout << "open " << filename_cnt << " failed." << std::endl;
//            cnt_infs.close();
//        }
//        while (cnt_infs.good()) {
//            cnt_infs >> src_ip >> dst_ip;
//            pkt_count ++;
//        }
//
//        //  use .groundtruth to calculate the true moment
//        ifstream true_infs(filename);
//        if (!true_infs) {
//            std::cout << "open " << filename << " failed." << std::endl;
//            true_infs.close();
//        }
//        while (true_infs.good()) {
//            true_infs >> src_ip >> true_card;
//            if (true_infs.eof()) break;
//            moments[0] += true_card * std::log(true_card);
//            for (int i = 1; i < 7; ++i) {
//                moments[i] += std::pow(true_card, i);
//            }
//        }
//    }
//
//    for (int i = 0; i < 7; ++i) {
//        cout << "L" << i << " moment: " << moments[i] << '\t' << moments[i] / pkt_count << endl;
//    }
//    cout << "packet number: " << pkt_count << endl;
//
//    string OUTPUT_DIR = "../../result/truth_moments.txt";
//    ofstream outf(OUTPUT_DIR);
//    for (int i = 0; i < 7; ++i) {
//        outf << moments[i] << ' ' << moments[i] / pkt_count << endl;
//    }

    //============================================================
    //  calculate the true cardinality distribution per file separately
    //============================================================
    vector<vector<double> > moments_per_file(130, vector<double>(7, 0));
    vector<unordered_map<double, double> > card_freq_per_file(130);
    vector<vector<pair<double, double> > > sorted_card_freq_per_file(130);
    vector<double> pkt_counts(130, 0);
    ifstream input_list(INPUT_FILE_LIST);
    string one_file;
    if (!input_list) {
        cout << "open " << INPUT_FILE_LIST << " failed!" <<endl;
        input_list.close();
    }
    int file_index = 0;
    while (input_list.good()) {
        input_list >> one_file;
        if (input_list.eof()) {
            break;
        }

        //  start to process one file
        string filename = DATA_DIR + one_file + "groundtruth";
        string filename_cnt = DATA_DIR + one_file + "uniq";

        double true_card;
        string src_ip;
        string dst_ip;

        //  use .uniq file to count total packet number
        ifstream cnt_infs(filename_cnt);
        if (!cnt_infs) {
            std::cout << "open " << filename_cnt << " failed." << std::endl;
            cnt_infs.close();
        }
        while (cnt_infs.good()) {
            cnt_infs >> src_ip >> dst_ip;
            if (cnt_infs.eof()) break;
            pkt_counts[file_index]++;
        }

        //  use .groundtruth to calculate the true moment and cardinality distribution
        ifstream true_infs(filename);
        if (!true_infs) {
            std::cout << "open " << filename << " failed." << std::endl;
            true_infs.close();
        }
        while (true_infs.good()) {
            true_infs >> src_ip >> true_card;
            if (true_infs.eof()) break;
            moments_per_file[file_index][0] += true_card * std::log(true_card);
            for (int i = 1; i < 7; ++i) {
                moments_per_file[file_index][i] += std::pow(true_card, i);
            }
            if (card_freq_per_file[file_index].count(true_card) == 0) {
                card_freq_per_file[file_index][true_card] = 1;
            } else {
                card_freq_per_file[file_index][true_card] += 1;
            }
        }
        for (auto item: card_freq_per_file[file_index]) {
            sorted_card_freq_per_file[file_index].emplace_back(make_pair(item.first, item.second));
        }
        sort(sorted_card_freq_per_file[file_index].begin(), sorted_card_freq_per_file[file_index].end(), CompareByKey<double, double>());

        cout << "File " << file_index+1 << ": " << filename_cnt << " Done." << endl;
        file_index++;
    }

    string moment_export_path = "../../result/truth_moments_pf.txt";
    string distri_export_path = "../../result/distribution_pf.txt";
    ofstream out_m(moment_export_path);
    ofstream out_d(distri_export_path);
    for (int i = 0; i < 130; ++i) {
        for (int j = 0; j < 7; ++j) {
            out_m << moments_per_file[i][j] << ' ' << moments_per_file[i][j] / pkt_counts[i] << endl;
        }
        out_m << endl;

        for (auto one_pair: sorted_card_freq_per_file[i]) {
            out_d << one_pair.first << ' ' << one_pair.second << endl;
        }
        out_d << endl;
    }

    return 0;
}
