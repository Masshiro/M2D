//
// Created by Masshiro on 2022/12/12.
//

#ifndef CARDINALITYMOMENTESTIMATOR_HEADTAIL_HPP
#define CARDINALITYMOMENTESTIMATOR_HEADTAIL_HPP
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <queue>
#include <string>
#include <bitset>
#include <unordered_map>
#include <random>
#include <iostream>

#include "../../src/utils/xxhash32.h"

namespace distribution{
    class HeadTail {
    private:
        double p_h;
        double p_t;
        int d_thr;
        std::unordered_map<std::string, bool> S_h;
        std::unordered_map<std::string, bool> S_t;
        std::unordered_map<std::string, int> ct_h;
        std::unordered_map<std::string, int> ct_t;

    public:
        HeadTail(double ph, double pt) {
            //  set two global parameters
            p_h = ph;
            p_t = pt;
        }

        int corrector(int r) {
            double num = 1-p_t - std::pow(1-p_t, r+1) - r * p_t * std::pow(1-p_t, r);
            double dec = p_t * (1-std::pow(1-p_t, r));
            return std::ceil(num / dec);
        }

        void update(std::string u, std::string v) {
            std::string temp;
            std::random_device rd;

            //  for u
            temp = u;
            if (S_h.count(temp) != 0) ct_h[temp] ++;
            else {
                XXHash32 hash32(rd());
                hash32.add(temp.c_str(), temp.length());
                double hash_val = double (hash32.hash() % 1000) / 1000;

                if (hash_val < p_h) {
                    S_h[temp] = true;
                    ct_h[temp] = 1;
                }
            }
            if (S_t.count(temp) != 0) ct_t[temp] ++;
            else {
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> distrib(0.0, 1.0);
                if (distrib(gen) <= p_t) {
                    S_t[temp] = true;
                    ct_t[temp] = 1;
                }
            }

            //  for v
            temp = v;
            if (S_h.count(temp) != 0) ct_h[temp] ++;
            else {
                XXHash32 hash32(rd());
                hash32.add(temp.c_str(), temp.length());
                double hash_val = double (hash32.hash() % 1000) / 1000;

                if (hash_val < p_h) {
                    S_h[temp] = true;
                    ct_h[temp] = 1;
                }
            }
            if (S_t.count(temp) != 0) ct_t[temp] ++;
            else {
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> distrib(0.0, 1.0);
                if (distrib(gen) <= p_t) {
                    S_t[temp] = true;
                    ct_t[temp] = 1;
                }
            }

        }

        void estimate() {
            //  construct C_h and C_t (line 1)
            std::unordered_map<int, bool> r_list_h;
            std::vector<int> r_list;
            std::unordered_map<int, int> C_h;
            std::unordered_map<int, int> C_t;
            for (auto & item: ct_h) {
                if (C_h.count(item.second) == 0) {
                    C_h[item.second] = 1;
                    r_list_h[item.second] = true;
                }
                else C_h[item.second] ++;
            }
            for (auto & item: ct_t) {
                if (C_t.count(item.second) == 0) {
                    C_t[item.second] = 1;
                    r_list_h[item.second] = true;
                }
                else C_t[item.second] ++;
            }
            for (auto & item: r_list_h) {
                r_list.emplace_back(item.first);
            }
            std::sort(r_list.begin(), r_list.end(), std::less<int>());


            //  construct C_t_tilde (line 2)
            std::unordered_map<int, int> C_t_tilde;
            for (auto r: r_list) {
                int cor_index = corrector(r);
                if (C_t.count(r-cor_index) != 0) C_t_tilde[r] = C_t[r-cor_index];
                else C_t_tilde[r] = 0;
            }

            //  construct g_h and g_t
            std::unordered_map<int, double> g_h;
            std::unordered_map<int, double> g_t;

        }
    };
}

#endif //CARDINALITYMOMENTESTIMATOR_HEADTAIL_HPP
