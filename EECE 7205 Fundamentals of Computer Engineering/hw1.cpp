//Ruiyue Wang NUID 001089745

#include <iostream>
#include <cmath>
#include <random>
using namespace std;

// Declare functions.
void MergeSort(int A[], int p, int r);
void Merge(int A[], int p, int q, int r);

int main(){
    // Create variable to store the size of input number array.
    int ArraySize;

    cout << "Please enter a number between 1 and 50 (not include 1):\n";
    cin >> ArraySize;

    // Generating random numbers.
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 100);

    // Creating array using dynamic memory.
    int* OriArray;

    OriArray = new int[ArraySize];

    //Generating random numbers in the array.
    for (int i=0; i<ArraySize; i++) {
        OriArray[i] = dis(gen);
    }

    // Calling Mergesort function to sort the original array.
    MergeSort(OriArray, 0, ArraySize);

    // Displaying the sorting result in screen.
    for (int j = 0; j < ArraySize; ++j) {
        cout << j + 1 << ' ' << OriArray[j] << '\n';
    }

    // Release memory.
    delete[] OriArray;
}

void MergeSort(int A[], int p, int r){
    int q;
    /* Using q as divided flag to divide the array until there is only one number in each array.
       And then calling the Merge function to merge the sorted arrays in to one. */
    if(p < r){
        q = floor((p+r)/2);
        MergeSort(A, p, q);
        MergeSort(A, q+1, r);
        Merge(A, p, q, r);
    }
}

void Merge(int A[], int p, int q, int r){
    /* Define the size of each half part of the original array,
    and then create the left half and right half part arrays with an extra position to store the sentinel number. */
    int n1 = q - p + 1;
    int n2 = r - q;
    int LeftHalf[n1+1];
    int RightHalf[n2+1];

    // Store the left half part array into LeftHalf and the right half into RightHalf.
    for (int i=0; i<n1; i++) {
        LeftHalf[i] = A[p + i];
    }

    for (int j=0; j<n2; j++) {
        RightHalf[j] = A[q + 1 + j];
    }

    // Add sentinel number into each array.
    LeftHalf[n1] = 101;
    RightHalf[n2] = 101;

    // Initializing the values of i and j.
    int i = 0;
    int j = 0;

    /* Since the LeftHalf and RightHalf arrays are sorted arrays to compare the smallest elements of them,
    and then add this element into the result array to get a whole sorted array. */
    for (int k=p; k<r+1; k++) {
        if (LeftHalf[i] <= RightHalf[j]) {
            A[k] = LeftHalf[i];
            i += 1;
        }

        else {
            A[k] = RightHalf[j];
            j += 1;
        }
    }
}

