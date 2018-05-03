#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <time.h>
using namespace std;
#define N 20242
#define NUM_F 47236
#define epsilon 0.00001
#define ITERATION 1000
double RATE=0.01;
double weights[NUM_F];
double gradient[NUM_F];
vector<string> datavevtor;

struct read_Node {
    int index;
    double value;
};

struct Instance {
    double features[NUM_F];
    int label;
};

double sigmoid(double x) {
    return 1.0/(1.0+exp(-x));
}

double predict(double x[]) {
    double logit=0.0;
    for (int i=0; i<NUM_F; i++) {
        logit+=weights[i]*x[i];
    }
    return sigmoid(logit);
}

void initialWeights() {
    srand(0);
    for (int i=0; i<NUM_F; i++) {
        double f=(double)rand()/RAND_MAX;
        weights[i]=-0.01+f*(0.02);
    }
    printf("initialWeights end.\n");
    printf("\n");
}

read_Node split_Element(string s) {
    int pos=s.find(":");
    read_Node ans;
    ans.index=stoi(s.substr(0,pos));
    ans.value=stod(s.substr(pos+1));
    return ans;
}

double norm(double w1[], double w2[]) {
    double sum=0.0;
    for (int i=0; i<NUM_F; i++) {
        double temp=w1[i]-w2[i];
        double r=temp*temp;
        sum+=r;
    }
    return sqrt(sum);
}

double sum_w(double w[]) {
    double sum=0.0;
    for (int i=0; i<NUM_F; i++) {
        sum+=w[i];
    }
    return sum;
}

int main() {
    ifstream input("./rcv1_train.binary");
    ofstream output("./logistic_gd.csv");
    output <<  "index" << "," <<  "Cross Entropy" << "," << "Accuracy" << endl;
    string line;
    srand(0);
    int epcho=0;
    int rd_index=0;
    int pre_right;
    while (getline(input,line)) {
        datavevtor.push_back(line);
    }
    printf("%lu\n", datavevtor.size());
    input.close();

    initialWeights();
    while (epcho<ITERATION) {
        for (int i=0; i<NUM_F; i++) {
            gradient[i]=0.0;
        }
        double loss;
        double old_w[NUM_F];
        for (int i=0; i<NUM_F; i++) {
            old_w[i]=weights[i];
        }
        pre_right=0;
        rd_index=0;
        loss=0.0;
        for (int t=0; t<datavevtor.size(); t++) {
            //read a line from dataset
            line=datavevtor[rd_index++];
            stringstream test(line);
            string s;
            Instance ins;
            for (int k=0; k<NUM_F; k++) {
                ins.features[k]=0.0;
            }
            while (getline(test,s,' ')) {
                if (s=="1" || s=="-1") {
                    ins.label=(s=="1"? 1:0);
                } else {
                    read_Node a=split_Element(s);
                    ins.features[a.index-1]=a.value;
                }
            }

            double o=0.0;

            for (int j=0; j<NUM_F; j++) {
                o+=weights[j]*ins.features[j];
            }
            double pre=sigmoid(o);

            if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) pre_right++;


            for (int j=0; j<NUM_F; j++) {
                gradient[j]+=(ins.label-pre)*ins.features[j];
            }

            loss+=-(ins.label*log2(pre)+(1-ins.label)*log2(1-pre));

        }
        for (int j=0; j<NUM_F; j++) {
            old_w[j]+=RATE*(gradient[j]);
        }

        // double dist=norm(weights,old_w);
        for (int i=0; i<NUM_F; i++) {
            weights[i]=old_w[i];
        }

        printf("%d : %f %f\n", epcho, loss, pre_right*1.0/N);
        output <<  epcho << "," <<  (loss) << "," << (pre_right*1.0/N) << endl;

        epcho++;
    }
    printf("%d\n", epcho);

    ifstream data("./rcv1_test.binary");
    int right=0;
    int NUM=0;

    while (NUM<50000 && getline(data,line)) {
        NUM++;
        stringstream test(line);
        string s;
        Instance ins;

        for (int k=0; k<NUM_F; k++) {
            ins.features[k]=0.0;
        }
        while (getline(test,s,' ')) {
            if (s=="1" || s=="-1") {
                ins.label=(s=="1"? 1:0);
            } else {
                read_Node a=split_Element(s);
                ins.features[a.index-1]=a.value;
            }
        }

        double o=0.0;
        for (int j=0; j<NUM_F; j++) {
            o+=weights[j]*ins.features[j];
        }
        double pre=sigmoid(o);
        if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) right++;
        if (NUM%10000==0) {
            printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
        }
    }
    data.close();

    printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
    output <<  right << "," <<  (NUM) << "," << (right*1.0/NUM) << endl;
    output.close();

    return 0;
}
