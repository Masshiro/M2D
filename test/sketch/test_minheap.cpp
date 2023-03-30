#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

std::vector<uint32_t> key_heap;
std::vector<int> index_heap;
std::vector<double> value_heap;

void assign(int i, int j) {
    key_heap[i] = key_heap[j];
    index_heap[i] = index_heap[j];
    value_heap[i] = value_heap[j];
}

void aton_push_heap_aux(int first, int holeIndex, int topIndex, double value, uint32_t key, int index) {
    int parent = (holeIndex - 1) / 2;//找出父节点
    while (holeIndex > topIndex && value_heap[first + parent] > value) { //尚未到顶端，并且父节点小于新值
//        value_heap[first + holeIndex] = value_heap[first + parent];
        assign(first + holeIndex, first + parent);
        holeIndex = parent; //更新holeIndex，继续向上层比较
        parent = (holeIndex - 1) / 2; //更新父节点
    }
    value_heap[first + holeIndex] = value;
    key_heap[first + holeIndex] = key;
    index_heap[first + holeIndex] = index;
}

void aton_push_heap(int first, int last) {
    aton_push_heap_aux(first, (last - first) - 1, 0, value_heap[last - 1], key_heap[last - 1], index_heap[last - 1]);
}

void aton_adjust_heap(int first, int holeIndex, int len, double value, uint32_t key, int index) {
    int topIndex = holeIndex;//顶点
    int secondChild = 2 * holeIndex + 2;//右子节点
    while (secondChild < len) {

        if (value_heap[first + secondChild] > value_heap[first + (secondChild - 1)])//左子节点比右子节点大
            secondChild--;//选择最大的子节点

//        value_heap[first + holeIndex] = value_heap[first + secondChild];//最大子节点为洞值
        assign(first + holeIndex, first + secondChild);
        holeIndex = secondChild;
        secondChild = 2 * (secondChild + 1);
    }
    if (secondChild == len) {
//        value_heap[first + holeIndex] = value_heap[first + (secondChild - 1)];
        assign(first + holeIndex, first + (secondChild - 1));
        holeIndex = secondChild - 1;
    }
    aton_push_heap_aux(first, holeIndex, topIndex, value, key, index);
}

void aton_make_heap(int first, int last) {
    if (last - first < 2)
        return;
    int len = last - first;
    int parent = (len - 2) / 2; //父节点，然后下沉

    while (true) {
        aton_adjust_heap(first, parent, len, value_heap[first + parent], key_heap[first + parent], index_heap[first + parent]);//parent下沉到合适的位置
        if (parent == 0) return;//构造完成
        parent--;
    }
}

void aton_pop_heap(int first, int last) {
    aton_adjust_heap(first, 0, last - 1 - first, value_heap[last - 1], key_heap[last - 1], index_heap[last - 1]);
}


int main() {
    key_heap.push_back(6);
    index_heap.push_back(6);
    value_heap.push_back(6);

    key_heap.push_back(2);
    index_heap.push_back(2);
    value_heap.push_back(2);

    key_heap.push_back(1);
    index_heap.push_back(1);
    value_heap.push_back(1);

    key_heap.push_back(4);
    index_heap.push_back(4);
    value_heap.push_back(4);

    key_heap.push_back(5);
    index_heap.push_back(5);
    value_heap.push_back(5);

    key_heap.push_back(9);
    index_heap.push_back(9);
    value_heap.push_back(9);

    key_heap.push_back(2);
    index_heap.push_back(2);
    value_heap.push_back(2);

    key_heap.push_back(3);
    index_heap.push_back(3);
    value_heap.push_back(3);

    aton_make_heap(0, value_heap.size());
    for (int i = 0; i < value_heap.size(); ++i) {
        std::cout << key_heap[i] << " " << index_heap[i] << " " << value_heap[i] << std::endl;
    }
    std::cout << std::endl;

    value_heap[1] = 7;
    aton_make_heap(0, value_heap.size());

    for (int i = 0; i < value_heap.size(); ++i) {
        std::cout << key_heap[i] << " " << index_heap[i] << " " << value_heap[i] << std::endl;
    }
    std::cout << std::endl;

    aton_pop_heap(0, value_heap.size());
    value_heap.pop_back();

    for (int i = 0; i < value_heap.size(); ++i) {
        std::cout << key_heap[i] << " " << index_heap[i] << " " << value_heap[i] << std::endl;
    }
    std::cout << std::endl;

    aton_pop_heap(0, value_heap.size());
    value_heap.pop_back();

    for (int i = 0; i < value_heap.size(); ++i) {
        std::cout << key_heap[i] << " " << index_heap[i] << " " << value_heap[i] << std::endl;
    }
    std::cout << std::endl;

    value_heap.emplace_back(5);

    aton_push_heap(0, value_heap.size());
    for (int i = 0; i < value_heap.size(); ++i) {
        std::cout << key_heap[i] << " " << index_heap[i] << " " << value_heap[i] << std::endl;
    }
    return 0;
}
