#include "../src/utils/CardinalityMap.hpp"
#include "spdlog/spdlog.h"


int main(int argc, char *argv[])
{
    CardinalityMap cm;
    cm.add(81,0.2);
    cm.add(52,0.6);
    cm.add(33,0.2);
    cm.add(52,0.1);
    cm.save_to_file("/tmp/cmsave1");
    spdlog::info("save");
    CardinalityMap tm;
    tm.read_from_file("/tmp/cmsave1");
    tm.save_to_file("/tmp/cmsave2");
    return 0;
}
