#include <iostream>
#include <math.h>
#include <time.h>
#include "mat.cpp"
#include <chrono>
using namespace std;
using namespace chrono;
//sigmoid
double sigmoid(double x){

    double y = 1.0 / (1.0 + exp(-x));
    return y;
}
//sigmoid
double sigmoid1(double y){

    double z = y*(1-y);
    return z;
}

double * rands(int length){
    srand((unsigned)time(NULL));
    double *retd = new double[length];
    for(int i=0;i<length;i++)retd[i]=double(1.0*(rand()%1000-500)/500);
    return retd;
}

class neu_net{
    private:
        mat ** weight;
        int length=0;
        mat ** b;

    public :
    neu_net(const int length,const int * pars){

        this->length=length-1;
        this->weight = new mat*[length-1];
        this->b = new mat*[length-1];

        for (int i=0;i<length-1;i++){
            double *doubpar=rands(pars[i+1]*pars[i]);
            mat* temp =new mat(pars[i+1],pars[i],doubpar);
            this->weight[i]=temp;
            delete doubpar;

            doubpar=rands(pars[i+1]*1);
            temp =new mat(pars[i+1],1,doubpar);
            this->b[i]=temp;
            delete doubpar;

        }
    }

    ~neu_net(){

        for(int i=0;i<this->length;i++){
            delete this->weight[i];
            delete this->b[i];
        }
        delete this->weight;
        delete this->b;
    }

    void show(){

        for( int i=0;i<this->length;i++){
            cout<<"layer "<<i<<" weights:"<<endl;
            this->weight[i]->show();
            cout<<endl;
            this->b[i]->show();
            cout<<endl;
        }

    }
    int getlength(){
        return this->length;
    }

    mat** getoutput(mat *input){
        mat* temp=input;
        mat ** output=new mat*[this->length];

        for(int i=0;i<this->length;i++){

            mat *tem =*(*(this->weight[i])*(temp))+this->b[i];
            double *dous = tem->todoubles();
            int len=tem->getrow()*tem->getcolumn();
            for(int j=0; j<len;j++){
                dous[j]=sigmoid(dous[j]);
            }
            output[i]=new mat(tem->getrow(),tem->getcolumn(),dous);
            temp=output[i];
        }
        return output;
    }


    void train(mat* x,mat *y,double r){

        mat ** h =this->getoutput(x);
        mat * g = new mat(*h[this->length-1]-y);

        for(int i=this->length-1;i>=0;i--){

            mat *th=h[i];
            double * tg=g->todoubles();
            double * tdb = th->todoubles();
            int len=(th->getcolumn())*(th->getrow());

            for(int j=0;j<len;j++){
                tg[j]=r*tg[j]*sigmoid1(tdb[j]);
            }
            g = new mat(g->getrow(),g->getcolumn(),tg);

            this->b[i]= *(this->b[i])-g;

            if(i==0){

                mat *dw=(*g)*(x->trans());
                this->weight[i]= *(this->weight[i]) - dw;
            }
            else{
                mat *dw =(*g)*(h[i-1]->trans());
                this->weight[i]= *(this->weight[i]) - dw;
            }

            g=(*this->weight[i]->trans())*(g);

        }

        for(int i=0;i<this->length;i++){
            delete h[i];
        }

    }

};

int main()
{
    double datset[14][4]={
        {0,1,0,0},
        {0,1,0,1},
        {0,1,1,0},
        {0,1,1,1},
        {1,0,0,0},
        {1,0,0,1},
        {1,0,1,0},
        {1,0,1,1},
        {0,0,0,0},
        {0,0,0,1},
        {1,1,0,0},
        {1,1,0,1},
        {1,1,1,0},
        {1,1,1,1}
    };

    int a[]={4,3,1};
    double b[14][1]={{0},{0},{0},{1},{0},{1},{1},{1},{1},{0},{0},{1},{1},{0}};
    neu_net n= neu_net(3,a);
    cout<<"start"<<endl;


    for(int j=0;j<10;j++) {
        auto start = system_clock::now();
        for(int i=0;i<1000;i++){
            for(int j=0;j<14;j++){
                mat inp =mat(4,1,datset[j]);
                mat y=mat(1,1,b[j]);
                n.train(&inp,&y,1);
            }
        }

        double cot=0;
        for(int j=0;j<14;j++){
            mat inp =mat(4,1,datset[j]);
            cot=cot+pow((b[j][0]-n.getoutput(&inp)[1]->todoubles()[0]),2);
        }
        auto end   = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout<<j<<" time "<<double(duration.count()) * microseconds::period::num / microseconds::period::den <<" var = "<< cot/14<<endl;
    }

    for(int i=0;i<14;i++){
        mat inp =mat(4,1,datset[i]);
        mat y=mat(1,1,b[i]);
        int r=n.getoutput(&inp)[1]->todoubles()[0]>0.5;
        cout<<"data "<<b[i][0]<<":"<<r<<endl;

    }

    return 0;
}
