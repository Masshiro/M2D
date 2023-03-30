#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

const vector<long long> col_list = {128, 256, 512, 1024, 2048, 4096, 8192};
#define ln2 0.693147180559945309417232121458176568075500134360255254120680009
#define ln2_n 710 // numerator of ln2
#define ln2_d 1024 // denominator of ln2
#define sqrt2 1.4142135623730950488016887242096980785696718753769480731766797
#define sqrt2_n 14142 // numerator of sqrt2
#define sqrt2_d 10000 // denominator of sqrt2


long long powerApprox(long long x, long long c) {
    long long q = x / c; // quotient
    long long r = x % c; // remainder
    long long res;

    if (2 * r < c) {
        res = (1 << q) * ((c + (ln2_n * r / ln2_d) + (ln2_n * ln2_n * r * r) / (2 * c * ln2_d * ln2_d)));
    } else {
        res = (sqrt2_n * (1 << q) *  (c + (ln2_n * (2 * r - c) / (2 * ln2_d)) + (ln2_n * ln2_n * (2 * r - c) * (2 * r - c) / (8 * c * ln2_d * ln2_d)))) / sqrt2_d;
    }

    return res;
}

int main() {

    for(long long x = 110; x < 46000; x = x + 100) {
        for(auto col:col_list) {
            double est_float = col * pow(2, x / (double)col);
            long long est_integer = powerApprox(x, col);
            double r_err = (est_integer - est_float) / est_float;
            cout << "x: " << x
                 << "\t"
                 << "c: " << col
                 << "\t"
                 << "float: " << est_float
                 << "\t"
                 << "integer: " << est_integer
                 << "\t"
                 << "r_err: " << r_err
                 << endl;
        }
    }

    return 0;
}
