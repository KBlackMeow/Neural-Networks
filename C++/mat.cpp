#include<iostream>
using namespace std;
class mat{
    private:

        double * data=nullptr;
        int row=0;
        int column=0;

        void free(){
            if(this->data==nullptr)return;
            else{
                delete this->data;
            }
        }
    public :

        mat(int row=0,int column=0, double * tdata=nullptr){

            this ->row=row;
            this ->column=column;
            this ->free();
            this ->data= new double [this->row*this->column];

            for (int i=0;i<this->row;i++){
                if(tdata!=nullptr){
                    for(int j=0;j<this->column;j++){
                        this->data[i*this->column+j]=tdata[this->column*i+j];
                    }
                }

            }



        }

        mat(const mat& tmat){

            this ->row=tmat.row;
            this ->column=tmat.column;
            this ->free();
            this ->data= new double [this->row*this->column];

            for (int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    this->data[i*this->column + j]=tmat.data[i*this->column + j];
                }
            }
        }

        mat(const mat *ptmat){

            this ->row=ptmat->row;
            this ->column=ptmat->column;
            this->free();
            this ->data= new double [this->row*this->column];

            for (int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    this->data[i*this->column + j]=ptmat->data[i*this->column + j];
                }
            }
        }

        ~mat(){
            delete this->data;
        }

        void show(){

            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    cout<<this->data[i*this->column + j]<<' ';
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

        mat* add(const mat* m){
            new mat(this->row,this->column);
            mat *temmat= new mat(this->row,this->column);

            if(this->row!=m->row||this->column!=m->column){
                cout<<"can not plus"<<endl;
                return  temmat;
            }
            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i*this->column + j]=this->data[i*this->column + j]+m->data[i*this->column + j];
                }
            }
            return temmat;
        }

        mat* sub (const mat* m){

            mat *temmat=new mat(this->row,this->column);
            if(this->row!=m->row||this->column!=m->column){
                cout<<"can not sub"<<endl;
                return  temmat;
            }

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i*this->column + j]=this->data[i*this->column + j]-m->data[i*this->column + j];
                }
            }

            return temmat;
        }

        mat* dot(const mat* m){

            mat * temmat= new mat(this->row,m->column);

            if(this->column!=m->row){
                cout<<"can not multiple"<<endl;
                return temmat;
            }

            for(int i=0;i<this->row;i++){
                for(int j=0;j<m->column;j++){

                    double temij=0;
                    for(int k=0;k<this->column;k++){
                        temij=temij+this->data[i*this->column + k]*m->data[k*m->column + j];
                    }
                    temmat->data[i*m->column + j]=temij;
                }
            }
            return temmat;
        }

        mat* dot(const double & m){

            mat *temmat= new mat(this->row,this->column);

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i*this->column + j]=this->data[i*this->column + j]*m;
                }
            }

            return temmat;
        }

        mat* exc (const double& m){

            mat * temmat= new mat(this->row,this->column);

            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[i*this->column + j]=this->data[i*this->column + j]/m;
                }
            }

            return temmat;
        }

        mat *trans(){

            mat *temmat= new mat(this->column,this->row);
            for(int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    temmat->data[j*this->row + i]=this->data[i*this->column + j];
                }
            }

            return temmat;
        }

        double * todoubles(){
            return this->data;
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
