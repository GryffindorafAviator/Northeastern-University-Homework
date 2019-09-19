//arguments.cc
//Passing arguments by value, by reference, and by pointer

#include <iostream>
using namespace std;

int squareByValue(int); // value pass
void squareByReference(int&); // reference pass
int squareByConstReference(const int&); // const reference pass
void squareByPointer(int*); // Pointer pass

int main() {
    int w{3}; // value to square using squareByValue
    int x{4}; // value to square using squareByReference
    int y{5}; // value to square using squareByConstReference
    int z{6}; // value to square using squareByPointer
    // demonstrate squareByValue
    cout << "w = " << w << " before squareByValue\n";
    cout << "Value returned by squareByValue: "<< squareByValue(w) << endl;
    cout << "w = " << w << " after squareByValue\n\n";
    // demonstrate squareByReference
    cout << "x = " << x << " before squareByReference\n";
    squareByReference(x);
    cout << "x = " << x << " after squareByReference\n\n";
    // demonstrate squareByConstReference
    cout << "y = " << y << " before squareByConstReference\n";
    cout << "Value returned by squareByConstReference: " << squareByConstReference(y) << endl;
    cout << "y = " << y << " after squareByConstReference\n\n";
    // demonstrate squareByPointer
    cout << "z = " << z << " before squareByPointer\n";
    squareByPointer(&z);
    cout << "z = " << z << " after squareByPointer\n";
}

// squareByValue multiplies number by itself, stores the
// result in number and returns the new value of number
int squareByValue(int number) {
    return number *= number; // caller's argument not modified
}

// squareByReference multiplies numberRef by itself and stores the result
// in the variable to which numberRef refers in function main
void squareByReference(int& numberRef) {
    numberRef *= numberRef; // caller's argument modified
}

// squareByConstReference returns the multiplication of numberRef by itself
int squareByConstReference(const int& numberCRef) {
    //numberCRef *= numberCRef; // Compilation error trying to modify a constant
    return numberCRef * numberCRef;
}

// squareByPointer multiplies number pointed to by numberPnt by itself
// and stores the result in the original variable
void squareByPointer(int* numberPnt) {
        *numberPnt = *numberPnt * *numberPnt; // caller's argument modified
}


