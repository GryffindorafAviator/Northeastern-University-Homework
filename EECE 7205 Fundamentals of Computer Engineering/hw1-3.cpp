#include <iostream>
#include <string>
using namespace std;
int ClassSize;

//Definition of the insertSort() function which has three arguments.
int insertSort(int grade[], string name[], int n) {
    //Using j to denote the key value which will be compared with these values before it.
    for (int j = 1; j < n; j++) {
        int keyG = grade[j];

        //Using keyN to store the jth value in the array of name, which can be used as "a pair with
        //the corresponding jth element in array of grades.
        string keyN = name[j];

        //Let k denotes the element just before the key value.
        int k = j - 1;

        //To compare the key value and the values before it.
        while (k >= 0 && grade[k] < keyG) {
            //As the question asking the descend order of the output list,
            //if the key value is greater than the value before it, the value will be switched,
            //otherwise it will stay in its original place.
            grade[k + 1] = grade[k];

            //The element in the array name will be changed following the array grade.
            name[k + 1] = name[k];

            //After one turn of comparision, k will switch to a position before.
            k -= 1;
        }

        //When encounter a value before the key value which is greater than the key value,
        //the program will jump out the while loop and place the key value at the position of the
        //value it encountered.
        grade[k + 1] = keyG;
        name[k + 1] = keyN;
    }

    //After sort the whole array, output the result.
    cout << "Their grades are: \n";
    for (int i = 0; i < n; i++) {
        cout << name[i] <<" "<< grade[i] << " \n";
    }
}

int main() {
    //Define the pointers for the name and grade array.
    int *gradesP;
    string *nameP;

    //The ClassSize will be used to creat the dynamic memory.
    cout << "What is the class size?\n";
    cin >> ClassSize;

    //Use dynamic memory to creat arrays.
    gradesP = new int[ClassSize];
    nameP = new string[ClassSize];

    //Store these input values into these arrays.
    for (int i = 0; i < ClassSize; i++) {
        cout << "What is the student's name?\n";
        cin >> nameP[i];
        cout << "What is his/her grade?\n";
        cin >> gradesP[i];
    }

    //Implementing sort and display the result.
    insertSort(gradesP, nameP, ClassSize);

    //Free up memory.
    delete[] nameP;
    delete[] gradesP;
}


