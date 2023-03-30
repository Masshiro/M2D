//
// Created by Masshiro on 2023/3/9.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "../../src/filter/MapImplFilter.hpp"
#include "gtest/gtest.h"
#include "fmt/core.h"

using namespace std;

TEST(map_filter, basic) {
    hierarchy::MapImplFilter<string> filter(5, 2);

    vector<string> ips = {"192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.2.1", "255.255.1.1", "213.12.11.12"};
    vector<double> cards = {0.14, 1.4, 2.567, 3.98, 4.345, 5.09};

    // condition 1: new flow id
    //  1.1) NOT full
    filter.insert_element_at_level_k(ips[0], cards[0], 1);
    filter.insert_element_at_level_k(ips[1], cards[1], 1);
    filter.insert_element_at_level_k(ips[2], cards[2], 1);
    filter.insert_element_at_level_k(ips[3], cards[3], 1);
    filter.insert_element_at_level_k(ips[4], cards[4], 1);
    cout << "filter[1] looks like: " << endl;
    filter.display_heap_at_level_k(1);

    //  1.2) full
    filter.insert_element_at_level_k(ips[5], cards[5], 1);
    cout << "filter[1] looks like: " << endl;
    filter.display_heap_at_level_k(1);

    // condition 2: old flow id
    //  2.1) new card is greater, meaning update is required
    filter.insert_element_at_level_k(ips[2], cards[4], 1);
    cout << "filter[1] looks like: " << endl;
    filter.display_heap_at_level_k(1);

    //  2.2) new card is not greater, so no update
    filter.insert_element_at_level_k(ips[3], cards[0], 1);
    cout << "filter[1] looks like: " << endl;
    filter.display_heap_at_level_k(1);
}