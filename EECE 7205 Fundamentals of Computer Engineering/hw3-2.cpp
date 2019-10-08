//Ruiyue Wang NUID 001089745

#include <iostream>
#include <cmath>
#include <random>
using namespace std;

// Define auxiliary functions.
int h1(int birth, int m);
int max(int a[], int size);
int min(int a[], int size);
double ave(int a[], int size);
double var(int a[], int size);

int main() {
    //Define hash table sizes and hash table arrays.
    int m1 = 64;
    int m2 = 66;
    int m3 = 67;
    int m4 = 61;
    int birthNum = 1000;

    int m1Array[m1];
    int m2Array[m2];
    int m3Array[m3];
    int m4Array[m4];
    int birthArray[birthNum];

    //Defaults the numbers in hash table arrays to 0.
    for (int i1=0; i1<m1; i1++) {
        m1Array[i1] = 0;
    }

//    for (int i5=0; i5<m1; i5++) {
//        cout << i5 << ' ' << m1Array[i5] << '\n';
//    }

    for (int i2=0; i2<m2; i2++) {
        m2Array[i2] = 0;
    }

    for (int i3=0; i3<m3; i3++) {
        m3Array[i3] = 0;
    }

    for (int i4=0; i4<m4; i4++) {
        m4Array[i4] = 0;
    }

    //Generating random birthdays.
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disM(1, 12);
    uniform_int_distribution<> disD(01, 28);
    uniform_int_distribution<> disY(00, 04);

    //Store birthdays into birthArray.
    for (int i=0; i<birthNum; i++) {
        birthArray[i] = disM(gen) * 10000 + disD(gen) * 100 + disY(gen);
    }

    //Hash birthdays into hash table.
    for (int j=0; j<birthNum; j++) {
//        cout << j + 1 << ' ' << birthArray[j] << '\n';

        //Calculate remainders for each birthday mod by different hash number.
        int rem1 = h1(birthArray[j], m1);
//        cout << "rem1 = " << rem1 << '\n';

        int rem2 = h1(birthArray[j], m2);
        int rem3 = h1(birthArray[j], m3);
        int rem4 = h1(birthArray[j], m4);

        //Count the number of collisions in each slot, and then store them into array respectively.
        for (int k=0; k<m1; k++) {
            if (rem1 == k) {
//                cout << "k = " << k << " m1Array[k] = " << m1Array[k] << '\n';
                m1Array[k] += 1;
                break;
            }
        }

        for (int l=0; l<m2; l++) {
            if (rem2 == l) {
                m2Array[l] += 1;
                break;
            }
        }

        for (int m=0; m<m3; m++) {
            if (rem3 == m) {
                m3Array[m] += 1;
                break;
            }
        }

        for (int n=0; n<m4; n++) {
            if (rem4 == n) {
                m4Array[n] += 1;
                break;
            }
        }
    }

//    int sum = 0;
//    for (int p=0; p<m4; p++) {
//        sum += m4Array[p];
//        cout << p << ' ' << m4Array[p] << '\n';
//    }
//
//    cout << sum << '\n';

    //Calculate the four statistic parameters for each hash table.
    int max1 = max(m1Array, m1);
    int min1 = min(m1Array, m1);
    double mean1 = ave(m1Array, m1);
    double variance1 = var(m1Array, m1);

    int max2 = max(m2Array, m2);
    int min2 = min(m2Array, m2);
    double mean2 = ave(m2Array, m2);
    double variance2 = var(m2Array, m2);

    int max3 = max(m3Array, m3);
    int min3 = min(m3Array, m3);
    double mean3 = ave(m3Array, m3);
    double variance3 = var(m3Array, m3);

    int max4 = max(m4Array, m4);
    int min4 = min(m4Array, m4);
    double mean4 = ave(m4Array, m4);
    double variance4 = var(m4Array, m4);

    //Display the results.
    cout << "Maximum value of hash-64: " << max1 << '\n';
    cout << "Minimum value of hash-64: " << min1 << '\n';
    cout << "Mean of hash-64: " << mean1 << '\n';
    cout << "Variance of hash-64: " << variance1 << '\n';
    cout << '\n';

    cout << "Maximum value of hash-66: " << max2 << '\n';
    cout << "Minimum value of hash-66: " << min2 << '\n';
    cout << "Mean of hash-66: " << mean2 << '\n';
    cout << "Variance of hash-66: " << variance2 << '\n';
    cout << '\n';

    cout << "Maximum value of hash-67: " << max3 << '\n';
    cout << "Minimum value of hash-67: " << min3 << '\n';
    cout << "Mean of hash-67: " << mean3 << '\n';
    cout << "Variance of hash-67: " << variance3 << '\n';
    cout << '\n';

    cout << "Maximum value of hash-61: " << max4 << '\n';
    cout << "Minimum value of hash-61: " << min4 << '\n';
    cout << "Mean of hash-61: " << mean4 << '\n';
    cout << "Variance of hash-61: " << variance4 << '\n';
    cout << '\n';
}

//Definition of hash function.
int h1(int birth, int m){
    return birth % m;
}

//Definition of choosing the maximum number in an array function.
int max(int a[], int size) {
    int maximum = a[0];

    for (int i=1; i<size; i++) {
        if (a[i] > maximum) {
            maximum = a[i];
        }
    }

    return maximum;
}


//Definition of choosing the minimum number in an array function.
int min(int a[], int size) {
    int minimum = a[0];

    for (int i=1; i<size; i++) {
        if (a[i] < minimum) {
            minimum = a[i];
        }
    }

    return minimum;
}

//Definition of mean calculation function.
double ave(int a[], int size) {
    double sum = 0;
    for (int i=0; i<size; i++) {
        sum += a[i];
    }

    return sum/size;
}

//Definition of variance calculation function.
double var(int a[], int size) {
    double sum = 0;
    double average = ave(a, size);

    for (int i=0; i<size; i++) {
        sum += pow(a[i]-average,2);
    }

    return sum/size;
}


