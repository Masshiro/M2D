#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include "fmt/core.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

class Statistics {
public:
    std::uint64_t flow_cnt;
    std::uint64_t memory_access;
    std::uint64_t rebase_cnt;
    std::uint64_t swap_in_cnt;
    std::uint64_t swap_out_cnt;
    std::uint64_t heap_update_cnt;
    std::uint64_t sketch_update_cnt;
    std::uint64_t heap_query_cnt;
    std::uint64_t sketch_query_cnt;
    std::uint64_t hash_cnt;

    std::string save_file_path;
    Statistics(std::string save_file_path) {
        this->save_file_path = save_file_path;
        flow_cnt = 0;
        memory_access = 0;
        rebase_cnt = 0;
        swap_in_cnt = 0;
        swap_out_cnt = 0;
        heap_update_cnt = 0;
        sketch_update_cnt = 0;
        heap_query_cnt = 0;
        sketch_query_cnt = 0;
        hash_cnt = 0;
    }

    void reset() {
        flow_cnt = 0;
        memory_access = 0;
        rebase_cnt = 0;
        swap_in_cnt = 0;
        swap_out_cnt = 0;
        heap_update_cnt = 0;
        sketch_update_cnt = 0;
        heap_query_cnt = 0;
        sketch_query_cnt = 0;
        hash_cnt = 0;
    }

    void addFlowCnt(std::uint32_t cnt = 1) { flow_cnt += cnt; }
    void addMemoryAccess(std::uint32_t cnt = 1) { memory_access += cnt; }
    void addRebaseCnt(std::uint32_t cnt = 1) { rebase_cnt += cnt; }
    void addSwapinCnt(std::uint32_t cnt = 1) { swap_in_cnt += cnt; }
    void addSwapoutCnt(std::uint32_t cnt = 1) { swap_out_cnt += cnt; }
    void addHeapUpdateCnt(std::uint32_t cnt = 1) { heap_update_cnt += cnt; }
    void addSketchUpdateCnt(std::uint32_t cnt = 1) { sketch_update_cnt += cnt; }
    void addHeapQueryCnt(std::uint32_t cnt = 1) { heap_query_cnt += cnt; }
    void addSketchQueryCnt(std::uint32_t cnt = 1) { sketch_query_cnt += cnt; }
    void addHashCnt(std::uint32_t cnt = 1) { hash_cnt += cnt; }


    void save_to_file(double pkt_cnt = 1) {
        std::ofstream ofs(save_file_path, std::ios::app);
        if (!ofs.is_open()) {
            std::cerr << "AccessStatics.hpp says: file error" << std::endl;
            exit(1);
        }

//        double avg = (double) memory_access / flow_cnt;
//        std::string output = fmt::format("rebase:{}, mem access:{}, swap-in:{}, swap-out:{}, heap update:{}, heap query:{}, sketch update:{}, sketch query:{}, hash:{}",
//                                         rebase_cnt, memory_access, swap_in_cnt, swap_out_cnt, heap_update_cnt, heap_query_cnt, sketch_update_cnt, sketch_query_cnt, hash_cnt);
        std::string output = fmt::format("rebase:{}, mem access:{}, swap-in:{}, swap-out:{}, heap update:{}, heap query:{}, sketch update:{}, sketch query:{}, hash:{}",
                                         double(rebase_cnt/pkt_cnt), double(memory_access/pkt_cnt), double(swap_in_cnt/pkt_cnt), double(swap_out_cnt/pkt_cnt), double(heap_update_cnt/pkt_cnt), double(heap_query_cnt/pkt_cnt), double(sketch_update_cnt/pkt_cnt), double(sketch_query_cnt/pkt_cnt), double(hash_cnt/pkt_cnt));
        std::cout << output << std::endl;
        ofs << output << std::endl;
        ofs.close();
    }
};

#endif
