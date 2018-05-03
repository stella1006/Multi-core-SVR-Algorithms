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
#define M N*2
#define CORE 4
#define NUM_F 47236
double weights[NUM_F];
double est_w[NUM_F];
double est_miu[NUM_F];
int rou[NUM_F];
#define epsilon 0.00001
#define ITERATION 200
#define T 30
double RATE=0.01;
struct timeval start, endd;
double t_=0.0;
string outname="./LG_svrg_lazy_"+to_string(CORE)+".csv";
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

double delta(vector<Node> x, double w[]) {
    double logit=0.0;
    for (int i=0; i<x.size(); i++) {
        logit+=w[x[i].index]*x[i].value;
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
        // if (NUM%10000==0) {
        //     printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
        // }
    }
    data.close();

    printf("%d/%d %f\n", right, NUM, right*1.0/NUM);
    output <<  right << "," <<  (NUM) << "," << (right*1.0/NUM) << endl;

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

void getLoss() {
    double loss=0.0;
    int right=0;
    // #pragma omp parallel for reduction(+:right, loss)
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
    gettimeofday(&endd,NULL);
    t_=(double)((endd.tv_sec*1000000+endd.tv_usec)-(start.tv_sec*1000000+start.tv_usec))/1000000;
    printf("loss %f %f %f\n", right*1.0/N, loss*1.0/N, t_);
    output << loss*1.0/N << endl;

}

void set_Miu() {
    for (int j=0; j<NUM_F; j++) {
        est_miu[j]=0.0;
    }
    // #pragma omp parallel for
    for (unsigned int i=0; i<N; i++) {
        Instance ins=allInstance[i];
        double sub_o=delta(ins.features,est_w);
        double pre=sigmoid(sub_o);

        for (int j=0; j<ins.features.size(); j++) {
            est_miu[ins.features[j].index]+=(ins.label-pre)*ins.features[j].value;
        }
    }
    // #pragma omp parallel for
    for (int j=0; j<NUM_F; j++) {
        est_miu[j]/=N;;
    }
}

int main() {
    int rd_index=0;
    read_data();
    printf("Read data: %lu\n", allInstance.size());
    printf("Rate: %f Iter: %d\n", RATE, ITERATION);
    output << RATE << "," << ITERATION << endl;


    gettimeofday(&start,NULL);
    initialWeights();
    initial_est_Weights();
    omp_set_num_threads(CORE);

    for (int iter=0; iter<ITERATION; iter++) {
        set_Miu();
        for (int i=0; i<NUM_F; i++) {
            weights[i]=est_w[i];
        }
        for (int i=0; i<NUM_F; i++) {
            rou[i]=0;
        }
        #pragma omp parallel for
        // for (int tid=0; tid<CORE; tid++) {
        //     int thread_num=omp_get_thread_num();
        for (int t=1; t<=M; t++) {
            double gradient[NUM_F];
            double sub_gra[NUM_F];
            rd_index=(int)rand()%N;
            Instance ins=allInstance[rd_index];

            for (int j=0; j<NUM_F; j++) {
                gradient[j]=0.0;
                sub_gra[j]=0.0;
            }
            double o1=delta(ins.features,weights);
            double pre1=sigmoid(o1);
            double o2=delta(ins.features,est_w);
            double pre2=sigmoid(o2);
            for (int j=0; j<ins.features.size(); j++) {
                int temp=t-rou[ins.features[j].index]-1;
                weights[ins.features[j].index]+=RATE*temp*est_miu[ins.features[j].index];
                gradient[ins.features[j].index]=(ins.label-pre1)*ins.features[j].value;
                sub_gra[ins.features[j].index]=(ins.label-pre2)*ins.features[j].value;
                weights[ins.features[j].index]+=RATE*(gradient[ins.features[j].index]-sub_gra[ins.features[j].index]);
                rou[ins.features[j].index]=t;
            }
        }
        // }
        printf("iter: %d ", iter);
        output<<iter<<",";
        getLoss();

        for (int j=0; j<NUM_F; j++) {
            est_w[j]=weights[j];
        }

    }

    output.close();
    return 0;
}
