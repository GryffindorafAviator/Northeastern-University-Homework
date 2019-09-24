//Ruiyue Wang NUID 001089745

#include <iostream>
using namespace std;

// Definition of SwapP
int SwapP(int* x, int* y){
    // Give the value of which x is pointing to to z.
    int z = *x;

    //Give the value of which y is pointing to to which x is pointing.
    *x = *y;

    //Give the value of z to which y is pointing.
    *y = z;
}

// Definition of SwapR
int SwaR(int& x, int& y){
    //Give the value of x to z.
    int z = x;

    //Give the value of y to x.
    x = y;

    //Give the value of z to y.
    y = z;
}

int main(){
    //Using some variables to test SwapP and SwapR.
    int a = 1, b = 2, c = 3, d = 4;

    cout << "Before Swap: " << a << " " << b << "\n";
    //Applying SwapP.
    SwapP(&a, &b);

    cout << "After using SwapP: " <<  a << " " << b << "\n";

    cout << "Before Swap: " << c << " " << d << "\n";
    //Applying SwapR.
    SwaR(c, d);

    cout << "After using SwapR: " <<  c << " " << d << "\n";
}

