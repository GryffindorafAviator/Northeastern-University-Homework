# include <iostream>
using namespace std;

int main ()
{
    int x = 1, y = 9;
    int *p1, *p2;
    p1 = &x;
    p2 = &y;

    // Define a new pointer p3.
    int *p3;

    //Exchange the addresses stored in p1, p2 and p3.
    p3 = p1;
    p1 = p2;
    p2 = p3;

    //Output the result.
    cout << *p1 << " and " << *p2 << endl; // Prints "9 and 1"
    return 0;
}




