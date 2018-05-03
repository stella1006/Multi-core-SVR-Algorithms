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
#define NUM_F 47236
#define epsilon 0.0000001
#define ITERATION 20242*200
#define NUM_SAMPLES 20242
#define CORE 4
double RATE=0.01;
double weights[NUM_F];
double gradient[NUM_F];
struct timeval start, endd;
double t_=0.0;
string outname="./LG_sgd_mt"+to_string(CORE)+".csv";
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

double norm(double w1[], double w2[]) {
    double sum=0.0;
    // #pragma omp parallel for reduction(+:sum)
    for (int i=0; i<NUM_F; i++) {
        double temp=w1[i]-w2[i];
        double r=temp*temp;
        sum+=r;
    }
    return sqrt(sum);
}

double getLoss() {
    double loss=0.0;
    int right=0;
    #pragma omp parallel for reduction(+:right, loss)
    for (unsigned int t=0; t<allInstance.size(); t++) {
        Instance ins=allInstance[t];

        double logit=0.0;
        for (int i=0; i<ins.features.size(); i++) {
            logit+=weights[ins.features[i].index]*ins.features[i].value;
        }
        double pre=sigmoid(logit);
        if ((pre>0.5 && ins.label==1) || (pre<0.5 && ins.label==0)) right++;
        loss+=-(ins.label*log2(pre)+(1-ins.label)*log2(1-pre));
    }
    // printf("acc %f ", right*1.0/NUM_SAMPLES);
    // output << right*1.0/NUM_SAMPLES << ",";
    // gettimeofday(&endd,NULL);
    // t_=(double)((endd.tv_sec*1000000+endd.tv_usec)-(start.tv_sec*1000000+start.tv_usec))/1000000;
    // printf("loss %f\n", loss/NUM_SAMPLES);
    // output << loss/NUM_SAMPLES << endl;
    return loss/NUM_SAMPLES;
}

int main() {
    output <<  "index" << "," <<  "Cross Entropy" << "," << "Accuracy" << endl;
    read_data();
    printf("Read data: %lu\n", allInstance.size());
    printf("Rate: %f Iter: %d\n", RATE, ITERATION/NUM_SAMPLES);

    int rd_index=0;
    int iter_index=0;

    gettimeofday(&start,NULL);
    initialWeights();
    omp_set_num_threads(CORE);
    #pragma omp parallel for
    for (int iter=0; iter<ITERATION ; iter++) {

        double old_w[NUM_F];
        double loss=0.0;
        for (int i=0; i<NUM_F; i++) {
            old_w[i]=weights[i];
        }
        RATE=8/(iter*1.0/50+1.0)+0.001;

        rd_index=(int)rand()%NUM_SAMPLES;
        Instance ins=allInstance[rd_index];
        double o=delta(ins.features);
        double pre=sigmoid(o);

        for (int j=0; j<ins.features.size(); j++) {
            gradient[ins.features[j].index]=(ins.label-pre)*ins.features[j].value;
        }

        for (int j=0; j<ins.features.size(); j++) {
            weights[ins.features[j].index]+=RATE*(gradient[ins.features[j].index]);
        }

        if (iter%NUM_SAMPLES==0) {
            double loss=getLoss();
            iter_index++;
            printf("iter %d %f\n", iter_index, loss);
            output << iter_index << "," << loss << endl;

        }
    }

    // test_model();
    output.close();

    return 0;
}
