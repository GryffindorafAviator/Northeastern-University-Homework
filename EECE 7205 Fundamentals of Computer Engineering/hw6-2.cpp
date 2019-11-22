// Ruiyue Wang NUID 001089745

#include <iostream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdio>
using namespace std;

vector<int> rod_length = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

vector<int> price(int length);
int cut_rod(vector<int> price, int length);
int memorized_cut_rod(vector<int> price, int length);
int memorized_cut_rod_aux(vector<int> price, int length, int rvn[]);

int main() {
    clock_t t_ri, t_dp;
    int length = 15;
    vector<int> rtl_prc = price(length);
    ostream_iterator<int> output{cout, " "};
//    vector<int> test_prc = {1, 5, 8, 9};
    cout << "Retail price: " << endl;
    copy(rtl_prc.cbegin(), rtl_prc.cend(), output);
    cout << endl;

//    Recursive implementation.
    t_ri = clock();
    int profit_ri = cut_rod(rtl_prc, length);
    t_ri = clock() - t_ri;
    cout << "Recursive implementation: " << endl;
    cout << "profit is " << profit_ri << endl;
    printf ("It took me %d clicks (%f seconds). \n", t_ri, ((float)t_ri) / CLOCKS_PER_SEC);

//    Dynamic programming.
    t_dp = clock();
    int profit_dp = memorized_cut_rod(rtl_prc, length);
    t_dp = clock() - t_dp;
    cout << "Dynamic programming: " << endl;
    cout << "profit is " << profit_dp << endl;
    printf ("It took me %d clicks (%f seconds). \n", t_dp, ((float)t_dp) / CLOCKS_PER_SEC);
    return 0;
}

//    Calculate retail price and store in a vector.
vector<int> price(int length) {
    static vector<int> prc;

    for (int i = 1; i <= length; i++) {
        if (i == 1) {
            prc.push_back(2);
        } else if (1 < i < length) {
            prc.push_back(floor(i * 2.5));
        } else {
            prc.push_back(2.5 * i - 1);
        }
    }
    return prc;
}

//    Recursive implementation.
int cut_rod(vector<int> price, int length) {
    if (length == 0) {
        return 0;
    }
    int revenue = -999;

    for (int i = 1; i <= length; i++) {
        revenue = max(revenue, price[i-1] + cut_rod(price, length - i));
    }
    return revenue;
}

//    Dynamic programming.
int memorized_cut_rod(vector<int> price, int length) {
    int* rvn = new int[length + 1] ;

    for (int i = 0; i <= length; i++) {
        rvn[i] = -999;
    }
//    ostream_iterator<int> output{cout, " "};
//    copy(rvn.cbegin(), rvn.cend(), output);
//    cout << endl;
    return memorized_cut_rod_aux(price, length, rvn);
}

//    Memory results.
int memorized_cut_rod_aux(vector<int> price, int length, int rvn[]) {
    int q;
//    cout << length - 1 << '\n' << rvn[length - 1] << endl;

    if (rvn[length] >= 0) {
        return rvn[length];
    }

    if (length == 0) {
        q = 0;
    } else {
        q = -999;
        for (int i = 1; i <= length; i++) {
            q = max(q, price[i - 1] + memorized_cut_rod_aux(price, length - i, rvn));
        }
    }

    rvn[length] = q;

    return q;
}