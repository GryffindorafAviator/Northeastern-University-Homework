#include <iostream>
#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <iterator>
using namespace std;

int main() {
    ostream_iterator<int> output{cout, " "};
    vector<int> iniInt = {14, 18, 6, 16, 8, 20, 4, 10, 12, 2};

//    Tasks for vector.
    vector<int> vecInt{iniInt.cbegin(), iniInt.cend()};

//    Task 3 Increase the value of each element in the vector by 40.
    for (int i = 0; i < vecInt.size(); i++) {
        vecInt[i] += 40;
    }
    cout << "The tasks' results for vector are: " << endl;
    copy(vecInt.cbegin(), vecInt.cend(), output);
    cout << endl;

//    Task 7 Check if the vector contains the value 44 and if so, find its location.
    for (int i = 0; i < vecInt.size(); i++) {
        if (vecInt.at(i) == 44) {
            cout << "The location of 44 is: " << i + 1 << endl;
            break;
        }
    }

//    Task 8 Find the location of the first element in the vector that is greater than 50.
    for (int i = 0; i < vecInt.size(); i++) {
        if (vecInt.at(i) > 50){
            cout << "The location of the first element greater than 50 is: " << i + 1 << endl;
            break;
        }
    }

//    Task 9 Determine whether all of the elements in the vector are greater than 45.
    bool flagGrtFF = true;

    for (int i = 0; i < vecInt.size(); i++) {
        if (vecInt[i] <= 45 ) {
            cout << "Not all the elements in the vector are greater than 45." << endl;
            flagGrtFF = false;
            break;
        }
    }

    if (flagGrtFF) {
        cout << "All the elements in the vector are greater than 45." << endl;
    }

    cout << endl;

//    Tasks for list.
    list<int> lsInt;
    lsInt.assign(iniInt.cbegin(), iniInt.cend());

//    Task 4 Increase the value of each element in the list by 30.
    list <int> :: iterator it;
    for (it = lsInt.begin(); it != lsInt.end(); it++) {
        *it += 30;
    }

    cout << "The tasks' results for list are: " << endl;
    copy(lsInt.cbegin(), lsInt.cend(), output);
    cout << endl;

    cout << endl;

//    Tasks for stack.
    stack<int> stkInt;

//    Task 5 Increase the value of each element in the stack by 20.

    for (int i = 0; i < iniInt.size(); i++) {
        stkInt.push(iniInt[i] + 20);
    }

    cout << "The tasks' results for stack are: " << endl;
    while (!stkInt.empty()) {
        cout << stkInt.top() << ' ';
        stkInt.pop();
    }
    cout << endl;

    cout << endl;

//    Tasks for priority queue.
    priority_queue<int> priInt;

//    Task 6 Increase the value of each element in the priority_queue by 10.
    for (int i = 0; i < iniInt.size(); i++) {
        priInt.push(iniInt[i] + 10);
    }

    cout << "The tasks' results for priority queue are: " << endl;

    while (!priInt.empty()) {
        cout << priInt.top() <<' ';
        priInt.pop();
    }
    cout << endl;

    cout << endl;

    return 0;
}
