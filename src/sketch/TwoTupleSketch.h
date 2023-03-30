#ifndef TWO_TUPLE_SKETCH_H
#define TWO_TUPLE_SKETCH_H

#include <cstdint>

class TwoTupleSketch
{
public:
    virtual bool offerFlow(const void* ptr_flow_id, uint64_t flow_id_len, const void* ptr_element_id, uint64_t element_id_len) = 0;
    virtual double decodeFlow(const void* ptr_flow_id, uint64_t flow_id_len) = 0;
    virtual void resetSketch() = 0;
    virtual void resetSeed(uint32_t new_seed) = 0;
};

#endif
