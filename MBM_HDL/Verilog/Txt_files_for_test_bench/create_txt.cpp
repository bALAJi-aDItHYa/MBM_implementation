#include<fstream>
#include<bits/stdc++.h>
using namespace std;

int main(){
	int B1, B2, prod;
	int i,j;
	fstream f_B1, f_B2, f_actual;
	f_B1.open("B1.txt", ios::out | ios::trunc);
	f_B2.open("B2.txt", ios::out | ios::trunc);
	f_actual.open("Actual.txt", ios::out | ios::trunc);

	if(!f_B1)
		cout<<"B1 not open"<<endl;

	if(!f_B1)
		cout<<"B2 not open"<<endl;

	if(!f_B1)
		cout<<"actual not open"<<endl;

	for(i=0;i<256;i++){
		B1=i;
		for(j=0;j<256;j++){
			B2=j;
			prod = B1*B2;
			f_B1<<B1<<endl;
			f_B2<<B2<<endl;
			f_actual<<prod<<endl;
		}
	}


	f_B1.close();
	f_B2.close();
	f_actual.close();

	return 0;
}