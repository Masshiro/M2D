//
// Created by Masshiro on 2023/2/27.
//
#include <iostream>
#include <random>
#include <bitset>

#include "gtest/gtest.h"

using namespace std;

static inline uint32_t get_leading_ones(uint32_t a, uint32_t max_len){
    uint32_t mask = 1;
    for (int i = 0; i != max_len; ++i){
        if ((a & mask) != 0){
            mask = mask<<1;
        } else {
            return i;
        }
    }
    return max_len;
}

void display_bits(bitset<32> bits) {
    for (int i = 0; i < 32; ++i) {
        cout << bits[i];
    }
    cout << endl;
}

TEST(leading_ones, basic) {
    vector<int> test_ints;
    int test_rounds = 20;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 2147483645);
    for (int i = 0; i < test_rounds; ++i) {
        test_ints.emplace_back(distrib(gen));
    }

    for(auto i: test_ints) {
        cout << "int = " << i << endl;
        cout << "\t bits: ";
        bitset<32> bits = i;
        display_bits(bits);
        cout << "\t leading ones: " << get_leading_ones(i, 32);
        cout << endl;
    }
}