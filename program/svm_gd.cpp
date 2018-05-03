#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;
#define N 20242
#define NUM_F 47236
#define C 1
#define CORE 4
#define ITERATION 200
double RATE=0.01;
double weights[NUM_F];
double gradient[NUM_F];

string outname="./svm_gd_"+to_string(CORE)+".csv";
ofstream output(outname);
struct Node {
    int index;
    double value;
};
struct Instance {
    vector<Node> features;
    int label;
};
vector<Instance> allInstance;

double max(double a, double b) {
    return a>b? a:b;
}

Node split_Element(string s) {
    int pos=s.find(":");
    Node ans;
    ans.index=stoi(s.substr(0,pos))-1;
    ans.value=stod(s.substr(pos+1));
    return ans;
}

Instance getInstance(string line) {
    stringstream test(line);
    string s;
    Instance ins;

    while (getline(test,s,' ')) {
        if (s=="1" || s=="-1") {
            ins.label=(s=="1"? 1:-1);
        } else {
            Node a=split_Element(s);
            ins.features.push_back(a);
        }
    }
    return ins;
}

void read_data() {
    ifstream input("./rcv1_train.binary");
    string line;
    while (getline(input,line)) {
        Instance ins=getInstance(line);
        allInstance.push_back(ins);
    }
    input.close();
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

void initialGradient() {
    #pragma omp parallel for
    for (int i=0; i<NUM_F; i++) {
        gradient[i]=weights[i];
    }
}


double getLoss() {
    double loss=0.0;
    string line;
    #pragma omp parallel for reduction(+:loss)
    for (int i=0; i<NUM_F; i++) {
        loss+=0.5*weights[i]*weights[i];
    }
    printf("w %f ", loss);
    #pragma omp parallel for reduction(+:loss)
    for (unsigned int t=0; t<allInstance.size(); t++) {
        Instance ins=allInstance[t];

        double logit=0.0;
        for (int i=0; i<ins.features.size(); i++) {
            logit+=weights[ins.features[i].index]*ins.features[i].value;
        }
        // logit+=weights[NUM_F];
        logit=1-logit*ins.label;

        if (logit>=0.0) loss+=C*logit;
    }

    return loss/N;
}

double delta(vector<Node> x) {
    double logit=0.0;

    for (int i=0; i<x.size(); i++) {
        logit+=weights[x[i].index]*x[i].value;
    }
    return logit;
}

void testModel() {
    ifstream data("./rcv1_test.binary");
    string line;
    int right=0;
    int NUM=0;

    while (getline(data,line)) {
        NUM++;
        Instance ins=getInstance(line);
        double del=delta(ins.features);

        if ((del>0 && ins.label==1) || (del<0 && ins.label==-1)) right++;

    }
    data.close();
    printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
    output << right << "," << NUM << "," << right*1.0/NUM << endl;
}

int main() {
    int pre_right;
    struct timeval start, end;
    // time_t start,ends;
    read_data();
    printf("Read data: %lu\n", allInstance.size());
    printf("C: %d Rate: %f\n", C, RATE);
    output << C << "," << RATE << "," << ITERATION << endl;
    output <<  "Loss" << "," << "Accuracy" << endl;

    gettimeofday(&start,NULL);
    initialWeights();
    for (int iter=1; iter<=ITERATION; iter++) {
        // start=clock();
        initialGradient();
        pre_right=0;
        // double total_loss=0.0;
        omp_set_num_threads(CORE);
        #pragma omp parallel for
        for (int tid=0; tid<CORE; tid++) {
            int thread_num=omp_get_thread_num();
            for (int t=thread_num*N/CORE; t<(thread_num+1)*N/CORE; t++) {
                Instance ins=allInstance[t];
                double del=delta(ins.features);

                if (ins.label*del>0) pre_right++;

                int temp=(ins.label*del<=1.0? 1:0);
                if (temp==1) {
                    for (int j=0; j<ins.features.size(); j++) {
                        gradient[ins.features[j].index]-=C*(ins.label*ins.features[j].value);
                    }
                    // gradient[NUM_F]-=C;
                }

            }
        }
        #pragma omp parallel for
        for (int j=0; j<NUM_F; j++) {
            weights[j]-=RATE*gradient[j];
        }
        // if (iter%100==0) {
        gettimeofday(&end,NULL);
        double t_=(double)((end.tv_sec*1000000+end.tv_usec)-(start.tv_sec*1000000+start.tv_usec))/1000000;
        printf("%d : ", iter);
        double loss=getLoss();
        double acc=pre_right*1.0/N;
        printf(" loss %f acc %f %d\n", loss, acc, iter);
        output << iter << "," << loss << endl;
        // }


    }
    // testModel();
    output.close();
    return 0;

}
