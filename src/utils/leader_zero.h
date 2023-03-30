#ifndef LEADER_ZERO_H
#define LEADER_ZERO_H

#include <stdint.h> 
uint32_t get_leader_zero(uint32_t hash_value);
int hllPatLen(uint32_t hash_value);

uint8_t count_leading_zeros(const void *ptr_flow_id, uint64_t flow_id_len, const void *ptr_element_id, uint64_t element_id_len, uint32_t seed);
uint8_t count_leading_zeros(const void *ptr_flow_id, uint64_t flow_id_len, uint32_t seed);
uint32_t get_leader_zero(uint32_t hash_value);

#endif // !LEADER_ZERO_H
