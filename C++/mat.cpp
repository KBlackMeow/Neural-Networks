#include<iostream>
using namespace std;
class mat{
    private:

        double ** data=nullptr;
        int row=0;
        int column=0;

        void free(){
            if(this->data==nullptr)return;
            else{
                for (int i=0;i<this->row;i++){
                delete this->data[i];
            }
            delete this->data;
            }
        }
    public :

        mat(int row=0,int column=0, double * tdata=nullptr){

            this ->row=row;
            this ->column=column;
            this->free();
            this ->data= new double *[this->row];

            for (int i=0;i<this->row;i++){
                this ->data[i]=new double [this->column];
                if(tdata!=nullptr)

                    for(int j=0;j<this->column;j++){
                        this->data[i][j]=tdata[this->column*i+j];

                    }

            }

        }

        mat(const mat& tmat){

            this ->row=tmat.row;
            this ->column=tmat.column;
            this->free();
            this ->data= new double *[this->row];

            for (int i=0;i<this->row;i++){
                this ->data[i]=new double [this->column];
                for(int j=0;j<this->column;j++){
                    this->data[i][j]=tmat.data[i][j];
                }
            }
        }

        mat(const mat *ptmat){

            this ->row=ptmat->row;
            this ->column=ptmat->column;
            this->free();
            this ->data= new double *[this->row];

            for (int i=0;i<this->row;i++){
                this ->data[i]=new double [this->column];
                for(int j=0;j<this->column;j++){
                    this->data[i][j]=ptmat->data[i][j];
                }
            }
        }

        ~mat(){
            for (int i=0;i<this->row;i++){
                delete this->data[i];
            }
            delete this->data;

        }

        void show(){

            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    cout<<this->data[i][j]<<' ';
                }

                cout<<endl;
            }
        }

        static mat* eye(int row){

            double *temd = new double[row*row];
            for (int i=0;i<row;i++){
                for( int j=0;j<row;j++){
                    if(i==j)temd[i*row+j]=1;
                    else temd[i*row+j]=0;
                }
            }
            mat *temmat = new mat(row,row,temd);
            delete temd;
            return temmat;
        }

        mat* operator+(const mat* m){
            mat *temmat= new mat(this->row,this->column);
            if(this->row!=m->row||this->column!=m->column){
                cout<<"can not plus"<<endl;
                return  temmat;
            }

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i][j]=this->data[i][j]+m->data[i][j];
                }
            }

            return temmat;
        }

        mat* operator-(const mat* m){
            mat *temmat=new mat(this->row,this->column);
            if(this->row!=m->row||this->column!=m->column){
                cout<<"can not sub"<<endl;
                return  temmat;
            }

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i][j]=this->data[i][j]-m->data[i][j];
                }
            }

            return temmat;
        }
        mat* operator*(const mat* m){

            mat * temmat= new mat(this->row,m->column);
            if(this->column!=m->row){
                cout<<"can not multiple"<<endl;
                return temmat;
            }

            for(int i=0;i<this->row;i++){
                for(int j=0;j<m->column;j++){

                    double temij=0;
                    for(int k=0;k<this->column;k++){
                        temij=temij+this->data[i][k]*m->data[k][j];
                    }
                    temmat->data[i][j]=temij;
                }
            }

            return temmat;
        }

        mat* operator*(const double & m){

            mat *temmat= new mat(this->row,this->column);

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i][j]=this->data[i][j]*m;
                }
            }

            return temmat;
        }
        mat* operator/(const double& m){

            mat * temmat= new mat(this->row,this->column);

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i][j]=this->data[i][j]/m;
                }
            }

            return temmat;
        }

        mat *trans(){
            mat *temmat= new mat(this->column,this->row);

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[j][i]=this->data[i][j];
                }
            }

            return temmat;
        }

        void setdata(int row=0,int column=0, double * tdata=nullptr){

            for (int i=0;i<this->row;i++){
                delete this->data[i];
            }
            delete this->data;

            this ->row=row;
            this ->column=column;
            this ->data= new double *[this->row];

            for (int i=0;i<this->row;i++){
                this ->data[i]=new double [this->column];
                if(tdata!=nullptr)
                    for(int j=0;j<this->column;j++){
                        this->data[i][j]=tdata[this->column*i+j];
                    }

            }

        }
        //注意释放内存 delete dous
        double * todoubles(){

            double *dous = new double[this->column*this->row];
            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    dous[i*this->column+j]=this->data[i][j];
                }
            }

            return dous;
        }
        int getrow(){
            return this->row;
        }
        int getcolumn(){
            return this->column;
        }

        mat inverse(){
            mat temmat= mat::eye(this->column);
            if(this->column!=this->row){
                cout<<"can not inverse"<<endl;
                return temmat;
            }

        }


};
