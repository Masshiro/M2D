#include "../src/sketch/HLL.hpp"
#include "../src/sketch/On_vHLL.hpp"
#include "../src/sketch/Ton_vHLL.hpp"
#include "../src/utils/CardinalityMap.hpp"
#include "../src/utils/leader_zero.h"
#include "../src/utils/xxhash32.h"
#include "fmt/core.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "gtest/gtest.h"

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

TEST(On_vHLL, inflation) {
    spdlog::set_level(spdlog::level::debug);
    std::size_t stage_num = 4, stage_row = 1024, stage_col = 128;
    const std::uint32_t flow_id = 0x1;
    On_vHLL sketch(stage_num, stage_row, stage_col, rd());
    std::vector<int> card_test = {20000, 40000, 60000, 80000, 100000};
    std::vector<int> a, b;

    for (auto t : card_test) {
        sketch.resetSketch();
        cout << "#####################################" << endl << "Testing for cardinality " << t << endl;

        int ini_val = rand() % 999998 + 1;
        a.clear();
        b.clear();

        /* Phase 1 */
        // Insert t elements
        double c_ant = 0;
        double c_new = 0;
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            // cout << c_new << " " << c_ant << endl;
            if (c_new > c_ant)
                a.push_back(x);
        }
        cout << "The Initial HLL estimate is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;

        // Test the set A
        sketch.resetSketch();
        // sketch.resetSeed(rd());
        for (auto x : a) {
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 1 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;


        /* Phase 2 */
        // Insert t elements
        c_ant = 0;
        c_new = 0;
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            if (c_new > c_ant) {
                a.push_back(x);
            }
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 2 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;


        /* Phase 3 */
        sketch.resetSketch();
        c_ant = 0;
        c_new = 0;
        for (auto it = a.rbegin(); it != a.rend(); ++it) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &(*it), sizeof(*it));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            if (c_new > c_ant) {
                b.push_back(*it);
            }
        }
        sketch.resetSketch();
        for (auto x : b) {
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
        }
        cout << "The length of list B is: " << b.size() << endl;
        cout << "The HLL estimate for B in phase 3 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;
    }
}

TEST(Ton_vHLL, inflation) {
    spdlog::set_level(spdlog::level::debug);
    std::size_t stage_num = 4, stage_row = 512, stage_col = 128;
    const std::uint32_t flow_id = 0x1;
    Ton_vHLL sketch(stage_num, stage_row, stage_col, rd());
    std::vector<int> card_test = {20000, 40000, 60000, 80000, 100000};
    std::vector<int> a, b;

    for (auto t : card_test) {
        sketch.resetSketch();
        cout << "#####################################" << endl << "Testing for cardinality " << t << endl;

        int ini_val = rand() % 999998 + 1;
        a.clear();
        b.clear();
        
        /* Phase 1 */
        // Insert t elements
        double c_ant = 0;
        double c_new = 0;
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            // cout << c_new << " " << c_ant << endl;
            if (c_new > c_ant) 
                a.push_back(x);
        }
        cout << "The Initial HLL estimate is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;
        
        // Test the set A
        sketch.resetSketch();
        // sketch.resetSeed(rd());
        for (auto x : a) {
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 1 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;


        /* Phase 2 */
        // Insert t elements
        c_ant = 0;
        c_new = 0;
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            if (c_new > c_ant) {
                a.push_back(x);
            }
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 2 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;
    

        /* Phase 3 */
        sketch.resetSketch();
        c_ant = 0;
        c_new = 0;
        for (auto it = a.rbegin(); it != a.rend(); ++it) {
            c_ant = c_new;
            sketch.offerFlow(&flow_id, sizeof(flow_id), &(*it), sizeof(*it));
            c_new = sketch.decodeFlow(&flow_id, sizeof(flow_id));
            if (c_new > c_ant) {
                b.push_back(*it);
            }
        }
        sketch.resetSketch();
        for (auto x : b) {
            sketch.offerFlow(&flow_id, sizeof(flow_id), &x, sizeof(x));
        }
        cout << "The length of list B is: " << b.size() << endl;
        cout << "The HLL estimate for B in phase 3 is: " << sketch.decodeFlow(&flow_id, sizeof(flow_id)) << endl;
    }
}

TEST(HLL, inflation) {
    std::size_t reg_num = 512;
    HLL sketch(reg_num, rd());
    std::vector<int> card_test = {20000, 40000, 60000, 80000, 100000};
    std::vector<int> a, b;

    for (auto t : card_test) {
        sketch.resetSketch();
        cout << "#####################################" << endl << "Testing for cardinality " << t << endl;
        int ini_val = rand() % 999998 + 1;
        a.clear();
        b.clear();

        /* Phase 1 */
        // Insert t elements
        double c_ant = 0;
        double c_new = 0; 
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&x, sizeof(x));
            c_new = sketch.decodeFlow();
            if (c_new > c_ant) {
                a.push_back(x);
            }
        }
        cout << "The Initial HLL estimate is: " << sketch.decodeFlow() << endl;
        
        // Test the set A
        sketch.resetSketch();
        // sketch.resetSeed(rd());
        for (auto x : a) {
            // cout << x << endl;
            sketch.offerFlow(&x, sizeof(x));
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 1 is: " << sketch.decodeFlow() << endl;


        /* Phase 2 */
        // Insert t elements
        c_ant = 0;
        c_new = 0;
        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            c_ant = c_new;
            sketch.offerFlow(&x, sizeof(x));
            c_new = sketch.decodeFlow();
            if (c_new > c_ant) {
                a.push_back(x);
            }
        }
        cout << "The length of list A is: " << a.size() << endl;
        cout << "The HLL estimate for A in phase 2 is: " << sketch.decodeFlow() << endl;
    

        /* Phase 3 */
        sketch.resetSketch();
        c_ant = 0;
        c_new = 0;
        for (vector<int>::reverse_iterator it = a.rbegin(); it != a.rend(); ++it) {
            c_ant = c_new;
            sketch.offerFlow(&(*it), sizeof(*it));
            c_new = sketch.decodeFlow();
            if (c_new > c_ant) {
                b.push_back(*it);
            }
        }
        sketch.resetSketch();
        for (auto x : b) {
            sketch.offerFlow(&x, sizeof(x));
        }
        cout << "The length of list B is: " << b.size() << endl;
        cout << "The HLL estimate for B in phase 3 is: " << sketch.decodeFlow() << endl;

    }
}

TEST(HLL, evasion) {
    std::size_t reg_num = 512;
    std::uint32_t temp_rd = rd();
    HLL sketch(reg_num, temp_rd);
    std::vector<int> card_test = {20000, 40000, 60000, 80000, 100000};
    std::vector<int> a, b;
    int fake_num = 0;

    for (auto t : card_test) {
        sketch.resetSketch();
        
        cout << "#####################################" << endl << "Testing for cardinality " << t << endl;
        int ini_val = rand() % 999998 + 1;
        a.clear();
        b.clear();

        for (int x = ini_val + 1; x < t + ini_val; ++x) {
            

            std::uint32_t hashval = XXHash32::hash(&x, sizeof(x), temp_rd);
            // std::cout << "outside hashval:" << hashval << std::endl;
            std::uint32_t new_val = get_leader_zero(hashval);
            // std::cout << "val:" << new_val << std::endl;
            if(new_val == 1) {
                fake_num = x;
                break;
            }
        }

        for (int x = 0; x < t; ++x) {
            sketch.offerFlow(&fake_num, sizeof(fake_num));
        }


        cout << "The HLL estimate is: " << sketch.decodeFlow() << endl;
    }
    
}