#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <omp.h>
#include <algorithm>
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;
#define N 20242
#define M N*2
#define NUM_F 47236
#define C 1
#define lambda 1e-4
#define ITERATION 250
#define CORE 4
double RATE=0.001;
double weights[NUM_F];
double est_w[NUM_F];
double est_miu[NUM_F];
int rou[NUM_F];
struct timeval start, endd;
double t_=0.0;

string outname="./svm_svrg_lazy_"+to_string(CORE)+".csv";
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
vector<Instance> testdata;

double max(double a, double b) {
    return a>b? a:b;
}

void initialWeights() {
    srand(0);
    #pragma omp parallel for
    for (int i=0; i<NUM_F; i++) {
        double f=(double)rand()/RAND_MAX;
        weights[i]=-0.01+f*(0.02);
    }
    printf("initialWeights end.\n");
    printf("\n");
}

void initial_est_Weights() {
    for (int i=0; i<NUM_F; i++) {
        est_w[i]=weights[i];
    }
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
        // datavevtor.push(line);
        Instance ins=getInstance(line);
        allInstance.push_back(ins);
    }
    input.close();
}

void read_test() {
    ifstream data("./rcv1_test.binary");
    string line;
    while (getline(data,line)) {
        Instance ins=getInstance(line);
        testdata.push_back(ins);
    }
    data.close();
}


void getLoss() {
    double loss=0.0;
    // double lo_=0.0;
    int right=0;
    string line;
    #pragma omp parallel for reduction(+:loss)
    for (int i=0; i<NUM_F; i++) {
        loss+=lambda*0.5*weights[i]*weights[i];
        // lo_+=0.5*est_miu[i]*est_miu[i];
    }
    printf("w %f ", loss);
    // #pragma omp parallel for reduction(+:right, loss)
    for (unsigned int t=0; t<N; t++) {
        Instance ins=allInstance[t];

        double logit=0.0;
        for (unsigned int i=0; i<ins.features.size(); i++) {
            logit+=weights[ins.features[i].index]*ins.features[i].value;
        }
        // logit+=weights[NUM_F];
        if (ins.label*logit>0.0) right++;
        logit=1-logit*ins.label;

        if (logit>=0.0) loss+=C*logit;
    }
    // gettimeofday(&endd,NULL);
    // t_=(double)((endd.tv_sec*1000000+endd.tv_usec)-(start.tv_sec*1000000+start.tv_usec))/1000000;
    printf("acc %f loss %f\n", right*1.0/N, loss/N);
    output << loss/N  << endl;

}

double delta(vector<Node> x, double w[]) {
    double logit=0.0;
    for (unsigned int i=0; i<x.size(); i++) {
        logit+=w[x[i].index]*x[i].value;
    }
    return logit;
}

void testModel() {
    int right=0;
    int NUM=testdata.size()+1;
    #pragma omp parallel for reduction(+:right)
    for (unsigned int i=0; i<testdata.size(); i++) {
        Instance ins=testdata[i];
        double del=delta(ins.features,weights);

        if ((del>0 && ins.label==1) || (del<0 && ins.label==-1)) right++;
    }

    printf("%d/%d %f\n\n", right, NUM, right*1.0/NUM);
    output << right << "," << NUM << "," << right*1.0/NUM << endl;
}

void set_Miu() {
    #pragma omp parallel for
    for (int j=0; j<NUM_F; j++) {
        est_miu[j]=0.0;
    }

    for (unsigned int i=0; i<N; i++) {
        Instance ins=allInstance[i];
        double del=delta(ins.features,est_w);
        est_miu[i]+=est_w[i];
        if (ins.label*del<1.0) {
            for (unsigned int j=0; j<ins.features.size(); j++) {
            // est_miu[ins.features[j].index]+=lambda*est_w[ins.features[j].index];
                est_miu[ins.features[j].index]+=-C*(ins.label*ins.features[j].value);
            }
        }
    }
    #pragma omp parallel for
    for (int j=0; j<NUM_F; j++) {
        est_miu[j]/=N;
    }
}

int main() {
    read_data();
    // read_test();
    printf("Read data: %lu\n", allInstance.size());
    printf("C: %d Rate: %lf Iter: %d\n", C, RATE, ITERATION);
    output << C << "," << RATE << "," << ITERATION << endl;

    gettimeofday(&start,NULL);
    initialWeights();
    initial_est_Weights();

    output << C << "," << CORE << endl;
    omp_set_num_threads(CORE);


    for (int iter=0; iter<ITERATION; iter++) {
        for (int i=0; i<NUM_F; i++) {
            weights[i]=est_w[i];
        }
        set_Miu();
        for (int i=0; i<NUM_F; i++) {
            rou[i]=0;
        }
        double gra1[NUM_F];
        double gra2[NUM_F];
        for (int j=0; j<NUM_F; j++) {
            gra1[j]=0.0;
            gra2[j]=0.0;
        }
        #pragma omp parallel for
        for (int tid=0; tid<CORE; tid++) {
            int thread_num=omp_get_thread_num();
            for (int t=thread_num*M/CORE+1; t<=(thread_num+1)*M/CORE; t++) {

                int rd_index=(int)rand()%N;
                Instance ins=allInstance[rd_index];

                double del1=delta(ins.features,weights);
                double del2=delta(ins.features,est_w);
                for (unsigned int i=0; i<ins.features.size(); i++) {
                    gra1[ins.features[i].index]=lambda*weights[ins.features[i].index];
                    gra2[ins.features[i].index]=lambda*est_w[ins.features[i].index];
                    if (ins.label*del1<1.0) {
                        gra1[ins.features[i].index]-=C*(ins.label*ins.features[i].value);
                    }
                    if (ins.label*del2<1.0) {
                        gra2[ins.features[i].index]-=C*(ins.label*ins.features[i].value);
                    }
                }

                for (unsigned int j=0; j<ins.features.size(); j++) {
                    int temp=t-rou[ins.features[j].index]-1;
                    weights[ins.features[j].index]-=RATE*temp*est_miu[ins.features[j].index];
                    weights[ins.features[j].index]+=-RATE*(gra1[ins.features[j].index]-gra2[ins.features[j].index]);
                    rou[ins.features[j].index]=t;
                }
            }
        }

        printf("iter: %d ", iter);
        output<<iter<<",";
        getLoss();

        for (int j=0; j<NUM_F; j++) {
            est_w[j]=weights[j];
        }

    }
    // testModel();
    output.close();
    return 0;

}
