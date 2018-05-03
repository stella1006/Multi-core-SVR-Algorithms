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
#define M N*2
#define NUM_F 47236
double weights[NUM_F];
double est_w[NUM_F];
double est_miu[NUM_F];
double gradient[NUM_F];
#define epsilon 0.00001
#define ITERATION 20
double RATE=0.1;
string outname="./LG_svrg_hog"+"_"+to_string(ITERATION)+".csv";
ofstream output(outname);
struct Instance {
    double features[NUM_F];
    int label;
};
vector<Instance> allInstance;

struct read_Node {
    int index;
    double value;
};

read_Node split_Element(string s) {
    int pos=s.find(":");
    read_Node ans;
    ans.index=stoi(s.substr(0,pos));
    ans.value=stod(s.substr(pos+1));
    return ans;
}

Instance getInstance(string line) {
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
    return ins;
}

void read_data() {
    ifstream input("./rcv1_train.binary");
    string line;
    while (getline(input,line)) {
        // datavevtor.push(line);
        Instance ins=getInstance(line);
        allInstance.push_back(ins);
    }
    input.close();
}

double sigmoid(double x) {
    return 1.0/(1.0+exp(-x));
}

void initial_est_Weights() {
    for (int i=0; i<NUM_F; i++) {
        est_w[i]=weights[i];
    }
}

double delta(double x[], double w[]) {
    double logit=0.0;
    for (int i=0; i<NUM_F; i++) {
        logit+=w[i]*x[i];
    }
    return logit;
}

void test_model() {
    ifstream data("./rcv1_test.binary");
    string line;
    int right=0;
    int NUM=0;

    while (NUM<30000 && getline(data,line)) {
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
        // if (NUM%10000==0) {
        //     printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
        // }
    }
    data.close();

    printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
    output <<  right << "," <<  (NUM) << "," << (right*1.0/NUM) << endl;

}

void set_Miu() {
    for (int j=0; j<NUM_F; j++) {
        est_miu[j]=0.0;
    }
    for (unsigned int i=0; i<N; i++) {
        Instance ins=allInstance[i];
        double sub_o=delta(ins.features,est_w);
        double pre=sigmoid(sub_o);

        for (int j=0; j<NUM_F; j++) {
            est_miu[j]+=(ins.label-pre)*ins.features[j];
        }
    }

    for (int j=0; j<NUM_F; j++) {
        est_miu[j]/=N;;
    }
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

double getLoss() {
    double loss=0.0;
    int right=0;
    #pragma omp parallel for reduction(+:right, loss)
    for (unsigned int t=0; t<allInstance.size(); t++) {
        Instance ins=allInstance[t];

        double logit=0.0;
        for (int i=0; i<NUM_F; i++) {
            logit+=weights[i]*ins.features[i];
        }
        double pre=sigmoid(logit);
        if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) right++;
        loss+=-(ins.label*log2(pre)+(1-ins.label)*log2(1-pre));
    }
    printf("acc %f ", right*1.0/N);
    output << right*1.0/N << ",";
    return loss/N;
}

int main() {

    int rd_index=0;
    double o=0.0;
    double pre=0.0;
    read_data();
    printf("Read data: %lu\n", allInstance.size());
    printf("Rate: %f Iter: %d\n", RATE, ITERATION);
    output << RATE << "," << ITERATION << endl;

    initialWeights();
    initial_est_Weights();
    for (int iter=1; iter<=ITERATION; iter++) {
        set_Miu();
        for (int i=0; i<NUM_F; i++) {
            weights[i]=est_w[i];
        }
        #pragma omp parallel for
        for (int t=1; t<=M; t++) {
            // initialGradient();
            double gradient[NUM_F];
            for (int i=0; i<NUM_F; i++) {
                gradient[i]=0.0;
            }

            rd_index=(int)rand()%N;
            Instance ins=allInstance[rd_index];

            o=delta(ins.features,weights);
            pre=sigmoid(o);
            for (int j=0; j<NUM_F; j++) {
                gradient[j]=(ins.label-pre)*ins.features[j];
            }

            o=delta(ins.features,est_w);
            pre=sigmoid(o);
            for (int j=0; j<NUM_F; j++) {
                gradient[j]+=-(ins.label-pre)*ins.features[j]+est_miu[j];
            }
            for (int j=0; j<NUM_F; j++) {
                weights[j]+=RATE*(gradient[j]);
            }

        }
        printf("iter: %d ", iter);
        double loss=getLoss();
        printf("loss %f\n", loss);
        output << loss << endl;

        for (int j=0; j<NUM_F; j++) {
            est_w[j]=weights[j];
        }
        if (iter%10==0) {
            test_model();
        }

    }
    output.close();
    return 0;
}
