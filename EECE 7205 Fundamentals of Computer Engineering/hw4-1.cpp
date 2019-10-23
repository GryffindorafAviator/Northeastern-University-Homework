//Ruiyue Wang NUID 001089745

#include <iostream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <string>

using namespace std;

//Definition of functions.
void insertionSort(int A[], int aSize);

void heapSort(int A[], int heapSize);
int left(int i);
int right(int i);
void maxHeapify(int A[], int i, int heapSize);
void buildMaxHeap(int A[], int heapSize);

void quickSort(int A[], int p, int r);
int partition(int A[], int p, int r);

void checkFun(int A[], int aSize, string name);
void writeRecord(string name, int moves, int comps);

//Define global variables.
int moves = 0;
int comps = 0;

int main() {
//Define array size.
    int aSize = 1000;
//Define arrays used in different sort methods.
    int BST_is[aSize];
    int AVG_is[aSize];
    int WST_is[aSize];
    int BST_hs[aSize];
    int AVG_hs[aSize];
    int WST_hs[aSize];
    int BST_qs[aSize];
    int AVG_qs[aSize];
    int WST_qs[aSize];
//Create best array.
    for (int i = 0; i < aSize; i++) {
        BST_is[i] = i;
        BST_hs[i] = i;
        BST_qs[i] = i;
//        cout << "BST_is " << i << ' ' << BST_is[i] << '\n';
//        cout << "BST_hs " << i << ' ' << BST_hs[i] << '\n';
//        cout << "BST_qs " << i << ' ' << BST_qs[i] << '\n';
    }
//Create average array.
    srand(time(nullptr));

    for (int i = 0; i < aSize; i++) {
        AVG_is[i] = rand() % 100000;
        AVG_hs[i] = AVG_is[i];
        AVG_qs[i] = AVG_is[i];
//        cout << "AVG_is " << i << ' ' << AVG_is[i] << '\n';
//        cout << "AVG_hs " << i << ' ' << AVG_hs[i] << '\n';
//        cout << "AVG_qs " << i << ' ' << AVG_qs[i] << '\n';
    }
//Create worst array.
    for (int i = aSize; i > 0; i--) {
        WST_is[aSize - i] = i;
        WST_hs[aSize - i] = i;
        WST_qs[aSize - i] = i;
    }
//    for (int i = 0; i < aSize; i++) {
//        cout << "WST_is " << i << ' ' << WST_is[i] << '\n';
//        cout << "WST_hs " << i << ' ' << WST_hs[i] << '\n';
//        cout << "WST_qs " << i << ' ' << WST_qs[i] << '\n';
//    }
//Calling insertion sort for best array and check its order.
    insertionSort(BST_is, aSize);
    checkFun(BST_is, aSize, "insertion sort of BST");
    writeRecord("insertion sort of BST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';

    moves = 0;
    comps = 0;
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//    for (int k = 0; k < aSize; k++) {
//        cout << k << ' ' << BST_is[k] << '\n';
//    }
//    cout << endl;
//Calling insertion sort for average array and check its order.
    insertionSort(AVG_is, aSize);
    checkFun(AVG_is, aSize, "insertion sort of AVG");
    writeRecord("insertion sort of AVG", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int k = 0; k < aSize; k++) {
//        cout << k << ' ' << AVG_is[k] << '\n';
//    }
//    cout << endl;
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling insertion sort for worst array and check its order.
    insertionSort(WST_is, aSize);
    checkFun(WST_is, aSize, "insertion sort of WST");
    writeRecord("insertion sort of WST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int k = 0; k < aSize; k++) {
//        cout << k << ' ' << WST_is[k] << '\n';
//    }
//    cout << endl;

//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling quick sort for best array and check its order.
    quickSort(BST_qs, 0, aSize-1);
    checkFun(BST_qs, aSize, "quick sort of BST");
    writeRecord("quick sort of BST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << BST_qs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling quick sort for average array and check its order.
    quickSort(AVG_qs, 0, aSize-1);
    checkFun(AVG_qs, aSize, "quick sort of AVG");
    writeRecord("quick sort of AVG", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << AVG_qs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling quick sort for worst array and check its order.
    quickSort(WST_qs, 0, aSize-1);
    checkFun(WST_qs, aSize, "quick sort of WST");
    writeRecord("quick sort of WST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << WST_qs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling heap sort for best array and check its order.
    heapSort(BST_hs, aSize);
    checkFun(BST_hs, aSize, "heap sort of BST");
    writeRecord("heap sort of BST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << BST_hs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling heap sort for average array and check its order.
    heapSort(AVG_hs, aSize);
    checkFun(AVG_hs, aSize, "heap sort of AVG");
    writeRecord("heap sort of AVG", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
    moves = 0;
    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << AVG_hs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//Calling heap sort for worst array and check its order.
    heapSort(WST_hs, aSize);
    checkFun(WST_hs, aSize, "heap sort of WST");
    writeRecord("heap sort of WST", moves, comps);
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';
//    moves = 0;
//    comps = 0;
//    for (int i = 0; i < aSize; i++) {
//        cout << WST_hs[i] << '\n';
//    }
//    cout << "number of moves: " << moves << '\n';
//    cout << "number of comps: " << comps << '\n';

    return 0;
}

//Definition of insertion.
void insertionSort(int A[], int aSize) {
//    cout << "value of moves and comps: " << moves << ' ' << comps << ' ' << '\n';
    for (int i = 1; i < aSize; i++) {
        int key = A[i];
        int j = i - 1;
//Compare key with each element before it.
        while (j>=0) {
            comps += 1;
            if (A[j]>key){
                moves += 1;
                A[j+1] = A[j];
                j -= 1;
            } else break;
        }
//Place key in its correct location.
        A[j+1] = key;
    }
}
//Define heap sort.
void heapSort(int A[], int heapSize) {
//    cout << "value of moves and comps: " << moves << ' ' << comps << ' ' << '\n';
//Build max heap.
    buildMaxHeap(A, heapSize);
//Switch root element and the last element.
    for (int i = heapSize - 1; i > 0; i--) {
        int exchange = A[0];
        A[0] = A[i];
        A[i] = exchange;
        moves += 3;

        heapSize -= 1;
//Rebuild the max heap.
        maxHeapify(A, 0, heapSize);
    }
}
//Define left child.
int left(int i) {
    return 2 * i + 1;
}
//Define right child.
int right(int i) {
    return 2 * i + 2;
}
//Define maxHeapity function.
void maxHeapify(int A[], int i, int heapSize) {
    int l = left(i);
    int r = right(i);
    int largest;
//Compare element i with its left child.
    comps += 1;
    if (l < heapSize && A[l] > A[i]) {
        largest = l;
    }

    else {
        largest = i;
    }
//Compare element i with its right child.
    comps += 1;
    if (r < heapSize && A[r] > A[largest]) {
        largest = r;
    }
//Place the biggest in the root of the subtree.
    if (largest != i) {
        int exchange = A[i];
        A[i] = A[largest];
        A[largest] = exchange;
        moves += 3;
//Rebuild the max heap.
        maxHeapify(A, largest, heapSize);
    }
}
//Build the max heap.
void buildMaxHeap(int A[], int heapSize) {
    for (int i = ceil(heapSize/2) - 1; i > -1; i--) {
        maxHeapify(A, i, heapSize);
    }
}
//Define the quick sort function.
void quickSort(int A[], int p, int r) {
//    cout << "value of moves and comps: " << moves << ' ' << comps << ' ' << '\n';
//Devide array into two parts.
    if (p < r) {
        int q = partition(A, p, r);
        quickSort(A, p, q-1);
        quickSort(A, q+1, r);
    }
}
//Define partition function.
int partition(int A[], int p, int r) {
    int x = A[r];
    int i = p - 1;
//Sort elements into two parts as it larger than or smaller than the pivot element.
    for (int j = p; j < r; j++) {
        comps += 1;
        if (A[j] <= x) {
            i += 1;
            int exchange1 = A[j];
            A[j] = A[i];
            A[i] = exchange1;
            moves += 3;
        }
    }
//Place pivot element into its correct place.
    int exchange2 = A[r];
    A[r] = A[i+1];
    A[i+1] = exchange2;
    moves += 3;

    return i + 1;
}
//Define check function.
void checkFun(int A[], int aSize, string name) {
    bool flag = true;
//    int a;
//    int b;
//    int numa;
//    int numb;
//To judge if A[i] is larger than A[i+1].
    for (int i = 0; i < aSize-1; i++) {
        if (A[i] > A[i+1]) {
//            a = A[i];
//            b = A[i+1];
//            numa = i;
//            numb = i + 1;
            flag = false;
            break;
        }
    }
//Using flag to display the result.
    if (flag) {
        cout << "The order of "<< name << " is correct." << endl;
    } else {
        cout << "The order of " << name << " is wrong." << endl;
    }
}
//Write the moves and comps into the txt file.
void writeRecord(string name, int moves, int comps) {
    ofstream records;

    records.open("sort.txt", ios::app);
        if (records.fail()) {
            cerr << "Error: Could not open output file.\n";
            exit(1);
        }

        records << "The number of moves of and comps of " << name << "： " << moves << " and " << comps << '\n';
        records << endl;

        records.close();
    }
