#include <iostream>
#include <cmath>
#include <cstring>
#include <fstream>
#include "ferns.h"

using namespace std;

double frand(double a, double b)
{
    double res = 1.0*rand()/RAND_MAX;
    return a + res*(b-a);
}

void GenData(int F, int N, double **X, int **Y)
{
    if(*X == NULL)
        *X = new double[N*F];
    if(*Y == NULL)
        *Y = new int[N];
    int g = N/4;
    for(int i = 0; i < g; i++)
    {//flat
        double a = frand(0, 1);
        for(int j = 0; j < F; j++)
            (*X)[i*F+j] = a;
        (*Y)[i] = 0;
    }
    for(int i = g; i < g*2; i++)
    {//go up
        for(int j = 0; j < F; j++)
        {
            double v = 1.0*j/(F-1);
            double a = frand(v-0.3, v+0.3);
            if(a < 0) a = 0;
            if(a > 1) a = 1;
            (*X)[i*F+j] = a;
        }
        (*Y)[i] = 1;
    }
    for(int i = g*2; i < g*3; i++)
    {//go down
        for(int j = 0; j < F; j++)
        {
            double v = 1.0*(F-1-j)/(F-1);
            double a = frand(v-0.3, v+0.3);
            if(a < 0) a = 0;
            if(a > 1) a = 1;
            (*X)[i*F+j] = a;
        }
        (*Y)[i] = 2;
    }
    for(int i = g*3; i < N; i++)
    {//go up and go down
        for(int j = 0; j < F/2; j++)
        {
            double v = 1.0 * j/(F/2-1);
            double a = frand(v-0.3, v+0.3);
            if(a < 0) a = 0;
            if(a > 1) a = 1;
            (*X)[i*F+j] = a;
        }
        for(int j = F/2; j < F; j++)
        {
            double v = 1.0 * (F-1-j)/(F/2 - 1);
            double a = frand(v-0.3, v+0.3);
            if(a < 0) a = 0;
            if(a > 1) a = 1;
            (*X)[i*F+j] = a;
        }
        (*Y)[i] = 3;
    }
}

int N, F, C, testN;
double *X, *testX;
int *Y, *testY;
int main()
{
    //Diff_Binary_feature dbf(5);
    //cout<<dbf.get_binary()<<endl;;

    C = 4;
    N = 1000;
    testN = 200;
    F = 10;
    GenData(F, N, &X, &Y);
    GenData(F, testN, &testX, &testY);
    cout<<"gendata over"<<endl;
    ofstream wf("test.out");
    wf << "train set has "<<N<<" entries,each entry has "<<F<<" feature\n";
    wf << "the train set matrix is\n";
    for(int i = 0; i < N; i++)
    {
        wf << "(";
        for(int j = 0; j < F-1; j++)
            wf << X[i*F+j] << ',';
        wf << X[i*F+F-1] << ")\n";
    }
    wf << "Right class is\n";
    for(int i = 0; i < N; i++)
        wf << Y[i] << endl;
    wf << endl;
    wf << "test set has "<<testN<<" entries,each entry has "<<F<<" feature\n";
    wf << "the test set matrix is\n";
    for(int i = 0; i < testN; i++)
    {
        wf << "(";
        for(int j = 0; j < F-1; j++)
            wf << testX[i*F+j] <<',';
        wf << testX[i*F+F-1] << ")" << endl;
    }
    wf << "Right class is\n";
    for(int i = 0; i < testN; i++)
        wf << testY[i] << endl;;
    wf << endl;
    wf << "train a single fern..." << endl;
    SingleFern sf(5);
    Diff_Binary_feature dbf(5);
    dbf.set_random(F, 0, 0);
    wf << "train single fern over, correct rate is "<<sf.train(X, Y, N, F, C, &dbf) << endl;
    wf << "single fern test.."<<endl;
    wf << "single fern test correct rate is " << sf.evaluate(testX, testY, testN, F) << endl;
    wf << endl;

    wf << "train a random ferns..." << endl;
    RandomFerns rf(10, 5);
    Diff_Binary_feature rdbf(50);
    rdbf.set_random(F, 0, 0);
    wf << "train random ferns over, correct rate is "<<rf.train(X, Y, N, F, C, &rdbf) << endl;
    wf << "random ferns test.."<<endl;
    wf << "random ferns test correct rate is " << rf.evaluate(testX, testY, testN, F) << endl;


    delete X;
    delete Y;
    delete testX;
    delete testY;

    return 0;
}
