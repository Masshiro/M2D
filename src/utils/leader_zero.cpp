#include "leader_zero.h"
#include "../utils/xxhash32.h"

#define HLL_Q 30

uint8_t count_leading_zeros(const void *ptr_flow_id, uint64_t flow_id_len, uint32_t seed) {
    uint32_t hash, bit, count;

    hash  = XXHash32::hash(ptr_flow_id, flow_id_len, seed);
    hash |= ((uint32_t)1<<HLL_Q);

    bit = 1;
    count = 1;
    while((hash & bit) == 0) {
        count++;
        bit <<= 1;
    }
    return count;
}

uint8_t count_leading_zeros(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len, uint32_t seed) {
    uint32_t hash, bit, count;

    XXHash32 value_hash_fun(seed);
    value_hash_fun.add(ptr_element_id, element_id_len);
    value_hash_fun.add(ptr_flow_id, flow_id_len);
    hash = value_hash_fun.hash();
    hash |= ((uint32_t)1<<HLL_Q);

    bit = 1;
    count = 1;
    while((hash & bit) == 0) {
        count++;
        bit <<= 1;
    }
    return count;
}

uint32_t get_leader_zero(uint32_t hash_value) {
    uint32_t mask = 1;
    uint32_t cnt = 1;
    // uint32_t(int)为4字节长度，sizeof(hash_value) * 8 共32bit
    for ( uint32_t i = 0; i < sizeof(hash_value) * 8; i++) {
        // 找到hash value二进制串最右的1的位置
        if ((mask & hash_value) != 0){
            break;
        }
        mask <<= 1;
        cnt++;
    }
    if (cnt >= 32) {
        cnt = 31;
    }
    // 返回值cnt的范围是1-31，可以在5bit内表示
    return cnt;
}

int hllPatLen(uint32_t hash) {
    uint32_t bit;
    int count;

    hash |= ((uint32_t)1<<HLL_Q);

    bit = 1;
    count = 1; /* Initialized to 1 since we count the "00000...1" pattern. */
    while((hash & bit) == 0) {
        count++;
        bit <<= 1;
    }
    return count;
}
