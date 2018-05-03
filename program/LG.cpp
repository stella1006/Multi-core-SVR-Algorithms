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
#define epsilon 0.00001
#define ITERATION 200
#define CORE 4
double RATE=0.001;
double weights[NUM_F];
double gradient[NUM_F];
string outname="./LG_"+to_string(CORE)+".csv";
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

double sigmoid(double x) {
    return 1.0/(1.0+exp(-x));
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
            ins.label=(s=="1"? 1:0);
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
        // datavevtor.push(line);
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
    for (int i=0; i<NUM_F; i++) {
        gradient[i]=0.0;
    }
}

double delta(vector<Node> x) {
    double logit=0.0;
    for (int i=0; i<x.size(); i++) {
        logit+=weights[x[i].index]*x[i].value;
    }
    return logit;
}

void test_model() {
    ifstream data("./rcv1_test.binary");
    string line;
    int right=0;
    int NUM=0;

    while (getline(data,line)) {
        NUM++;
        stringstream test(line);
        string s;
        Instance ins;

        while (getline(test,s,' ')) {
            if (s=="1" || s=="-1") {
                ins.label=(s=="1"? 1:0);
            } else {
                Node a=split_Element(s);
                ins.features.push_back(a);
            }
        }

        double o=0.0;
        for (int j=0; j<ins.features.size(); j++) {
            o+=weights[ins.features[j].index]*ins.features[j].value;
        }
        double pre=sigmoid(o);
        if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) right++;

    }
    data.close();

    printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
    output <<  right << "," <<  (NUM) << "," << (right*1.0/NUM) << endl;

}

int main() {
    struct timeval start, end;

    output <<  "index" << "," <<  "Cross Entropy" << "," << "Accuracy" << endl;
    read_data();
    printf("Read data: %lu\n", allInstance.size());
    printf("Rate: %f Iter: %d\n", RATE, ITERATION);

    int epcho=0;
    int pre_right;
    double t_=0.0;

    // gettimeofday(&start,NULL);
    initialWeights();
    while (epcho<ITERATION) {
        initialGradient();
        pre_right=0;
        double total_loss=0.0;
        omp_set_num_threads(CORE);
        #pragma omp parallel for reduction(+:total_loss)
        for (int tid=0; tid<CORE; tid++) {
            double loss=0.0;
            int thread_num=omp_get_thread_num();
            for (int t=thread_num*N/CORE; t<(thread_num+1)*N/CORE; t++) {
                Instance ins=allInstance[t];
                double o=delta(ins.features);
                double pre=sigmoid(o);
                if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) pre_right++;

                for (int j=0; j<ins.features.size(); j++) {
                    gradient[ins.features[j].index]+=(ins.label-pre)*ins.features[j].value;
                }
                loss+=-(ins.label*log2(pre)+(1-ins.label)*log2(1-pre));
            }
            total_loss+=loss;
        }


        #pragma omp parallel for
        for (int j=0; j<NUM_F; j++) {
            weights[j]+=RATE*(gradient[j]);
        }
        // gettimeofday(&end,NULL);
        // t_=(double)((end.tv_sec*1000000+end.tv_usec)-(start.tv_sec*1000000+start.tv_usec))/1000000;
        printf("%d : %f\n", epcho, total_loss/N);
        output <<  epcho << "," <<  (total_loss/N) << endl;
        epcho++;
    }
    printf("%d\n", epcho);

    // test_model();
    output.close();

    return 0;
}
