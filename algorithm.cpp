#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include "algorithm.h"
#include "svm.h"

using namespace std;
//�ڴ����
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* ��װsvm                                                                     */
/************************************************************************/

//************************************
// ��    ��: ���캯��
// ��    ��: CxLibSVM
// �� �� ��: CxLibSVM::CxLibSVM
// ����Ȩ��: public 
// �� �� ֵ: 
// �� �� ��:
//************************************
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include "algorithm.h"
#include "svm.h"

using namespace std;
//�ڴ����
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* ��װsvm                                                                     */
/************************************************************************/


libsvm::libsvm()
{
	model_ = NULL;
}


libsvm::~libsvm()
{
	free_model();
}

//���ݹ�һ������
void	libsvm::noemal(int* basicData, int* backgroundData, int normalLength, double* normalData, string light)
{
	double data[256];
	double minValue;
	double maxValue;
	//double normalValue;

	if (light == "absorbinglight")
	{
		//printf("���չ�");

		//��ȡ�������
		for (int i = 0; i < normalLength; i++)
		{
			data[i] = basicData[i] / backgroundData[i];
		}
	}
	else if (light == "fluorescence")
	{
		//printf("ӫ��");

		//��ȡ�ۼ�����
		for (int i = 0; i < normalLength; i++)
		{
			data[i] = basicData[i] - backgroundData[i];
		}
	}


	//��ȡ�����Сֵ
	minValue = data[0];
	for (int i = 0; i < normalLength; i++)
	{
		if (data[i] < minValue)
			minValue = data[i];
	}
	maxValue = data[0];
	for (int i = 0; i < normalLength; i++)
	{
		if (data[i] > maxValue)
			maxValue = data[i];
	}

	//���й�һ��
	for (int i = 0; i < normalLength; i++)
	{
		normalData[i] = double((data[i] - minValue)) / double((maxValue - minValue));
	}
}




// ӫ�⽨ģ���ݴ���(��ѡ�����ݷ�Χ)
void libsvm::fluorescenceTrainHandl(int train[][256], int trainLength, int exposeTimeData[256], int startIndex, int endIndex, vector<vector<double>>& train_X, int num, vector<double>& train_Y)
{
	int cutData[256];
	int cutExposeData[256];
	for (int i = 0; i < trainLength; i++)
	{

		for (int j = 0, k= startIndex; k<= endIndex; j++,k++)
		{
			cutData[j] = train[i][k];
			cutExposeData[j] = exposeTimeData[k];
		}
		noemal(cutData, cutExposeData, endIndex-startIndex+1, normalData, "fluorescence");

		vector<double> rx;
		for (int m = 0; m < endIndex-startIndex+1; m++)
		{
			rx.push_back(normalData[m]);
			//cout << normalData[m] << " ";
		}
		train_X.push_back(rx);
		train_Y.push_back(num);
	}
}





// ӫ��������ݴ���
void libsvm::fluorescenceTestHandl(int test[256], int exposeTimeData[256], int startIndex, int endIndex, vector<double>& test_X)
{
	int cutData[256];
	int cutExposeData[256];
	for (int j = 0, k = startIndex; k <= endIndex; j++, k++)
	{
		cutData[j] = test[k];
		cutExposeData[j] = exposeTimeData[k];
	}
	noemal(cutData, cutExposeData, endIndex - startIndex + 1, normalData, "fluorescence");

	for (int j = 0; j < endIndex - startIndex + 1; j++)
	{
		test_X.push_back(normalData[j]);
		//cout << normalData[j] << " ";
	}
	//cout << endl;

}



// ���չ⽨ģ���ݴ���
void libsvm::absorbinglightTrainHandl(int train[][256], int trainLength, int exposeTimeData[256], int startIndex, int endIndex, vector<vector<double>>& train_X, int num, vector<double>& train_Y)
{
	int cutData[256];
	int cutExposeData[256];
	for (int i = 0; i < trainLength; i++)
	{

		for (int j = 0, k = startIndex; k <= endIndex; j++, k++)
		{
			cutData[j] = train[i][k];
			cutExposeData[j] = exposeTimeData[k];
		}
		noemal(cutData, cutExposeData, endIndex - startIndex + 1, normalData, "absorbinglight");

		vector<double> rx;
		for (int m = 0; m < endIndex - startIndex + 1; m++)
		{
			rx.push_back(normalData[m]);
			//cout << normalData[m] << " ";
		}

		train_X.push_back(rx);
		train_Y.push_back(num);
	}
}



// ���չ�������ݴ���
void libsvm::absorbinglightTestHandl(int test[256], int exposeTimeData[256], int startIndex, int endIndex, vector<double>& test_X)
{
	int cutData[256];
	int cutExposeData[256];
	for (int j = 0, k = startIndex; k <= endIndex; j++, k++)
	{
		cutData[j] = test[k];
		cutExposeData[j] = exposeTimeData[k];
	}
	noemal(cutData, cutExposeData, endIndex - startIndex + 1, normalData, "absorbinglight");

	for (int j = 0; j < endIndex - startIndex + 1; j++)
	{
		test_X.push_back(normalData[j]);
	}
}



void libsvm::train(const vector<vector<double>>& x, const vector<double>& y, string model_path)
{
	if (x.size() == 0)return;

	//�ͷ���ǰ��ģ��
	free_model();

	/*��ʼ��*/
	long	len = x.size();
	long	dim = x[0].size();
	long	elements = len * dim;

	//������ʼ�����������������������޸ļ���
	// Ĭ�ϲ���
	param.svm_type = C_SVC;		//�㷨����
	param.kernel_type = LINEAR;	//�˺�������
	param.degree = 3;	//����ʽ�˺����Ĳ���degree
	param.coef0 = 0;	//����ʽ�˺����Ĳ���coef0
	param.gamma = 0.5;	//1/num_features��rbf�˺�������
	param.nu = 0.5;		//nu-svc�Ĳ���
	param.C = 10;		//������ĳͷ�ϵ��
	param.eps = 1e-3;	//��������
	param.cache_size = 100;	//�����ڴ滺�� 100MB
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;	//1��ʾѵ��ʱ���ɸ���ģ�ͣ�0��ʾѵ��ʱ�����ɸ���ģ�ͣ�����Ԥ���������������ĸ���
	param.nr_weight = 0;	//���Ȩ��
	param.weight = NULL;	//����Ȩ��
	param.weight_label = NULL;	//���Ȩ��


	//ת������Ϊlibsvm��ʽ
	prob.l = len;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node*, prob.l);
	x_space = Malloc(struct svm_node, elements + len);
	int j = 0;
	for (int l = 0; l < len; l++)
	{
		prob.x[l] = &x_space[j];
		for (int d = 0; d < dim; d++)
		{
			x_space[j].index = d + 1;
			x_space[j].value = x[l][d];
			j++;
		}
		x_space[j++].index = -1;
		prob.y[l] = y[l];
	}

	/*ѵ��*/
	model_ = svm_train(&prob, &param);

	cout << "********************" << endl;
	cout << model_ << endl;
	cout << "********************" << endl;

	save_model(model_path);			// ����ģ��
}




	//************************************
	// ��    ��: Ԥ����������������͸���
	// ��    ��: predict
	// �� �� ��: CxLibSVM::predict
	// ����Ȩ��: public 
	// ��    ��: const vector<double> & x	����
	// ��    ��: double & prob_est			�����Ƶĸ���
	// �� �� ֵ: double						Ԥ������
	// �� �� ��:
	//************************************
	//int predict(const vector<double>& x,double& prob_est)
	//{
	//	//����ת��
	//	svm_node* x_test = Malloc(struct svm_node, x.size()+1);
	//	for (unsigned int i=0; i<x.size(); i++)
	//	{
	//		x_test[i].index = i;
	//		x_test[i].value = x[i];
	//	}
	//	x_test[x.size()].index = -1;
	//	double *probs = new double[model_->nr_class];//�洢���������ĸ���
	//	//Ԥ�����͸���
	//	int value = (int)svm_predict_probability(model_, x_test, probs);
	//	for (int k = 0; k < model_->nr_class;k++)
	//	{//����������Ӧ�ĸ���
	//		if (model_->label[k] == value)
	//		{
	//			prob_est = probs[k];
	//			break;
	//		}
	//	}
	//	delete[] probs;
	//	return value;
	//}

int libsvm::predict(string model_path, const vector<double>& x)
{
	load_model(model_path);		// ����ģ��

	//����ת��
	svm_node* x_test = Malloc(struct svm_node, x.size() + 1);
	for (unsigned int i = 0; i < x.size(); i++)
	{
		x_test[i].index = i;
		x_test[i].value = x[i];
	}
	x_test[x.size()].index = -1;
	double* probs = new double[model_->nr_class];//�洢���������ĸ���

	//Ԥ�����͸���
	int value = (int)svm_predict_probability(model_, x_test, probs);

	delete[] probs;
	return value;
}




	//************************************
	// ��    ��: ����svmģ��
	// ��    ��: load_model
	// ����Ȩ��: public 
	// ��    ��: string model_path	ģ��·��
	// �� �� ֵ: int 0��ʾ�ɹ���-1��ʾʧ��
	// �� �� ��:
	//************************************
int libsvm::load_model(string model_path)
{
	//�ͷ�ԭ����ģ��
	free_model();
	//����ģ��
	model_ = svm_load_model(model_path.c_str());
	if (model_ == NULL)return -1;
	return 0;
}

int libsvm::save_model(string model_path)
{
	int flag = svm_save_model(model_path.c_str(), model_);
	return flag;
}


void libsvm::free_model()
{
	if (model_ != NULL)
	{
		svm_free_and_destroy_model(&model_);
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
	}
}

double  libsvm::getDataMean(int* newData)
{
	double mean = 0;
	for (int i = 38; i < 256; i++)
	{
		mean += newData[i];
	}
	return (int)mean / 218;
}


double libsvm::getDataSum1(double xMean, double yMean, int*  xData, int* yData)
{
	double sum1=0;
	for (int i=38;i<256;i++) 
	{
		sum1 +=  (xData[i] - xMean) * (yData[i] - yMean);  
	}
	return sum1;
}

double libsvm::getDataSum2( int* Data, double Mean )
{
	double sum2=0;
	for (int i = 38; i < 256; i++)
	{
		sum2 += pow((Data[i] - Mean), 2);
	}
	return sum2;
}

//Ƥ��ɭϵ���б�����ԭʼ���ݣ�
double  libsvm::pearsonCorrelation(int* newData, int* oldData)
{
	double xMean;
	double yMean;
	double sum1;
	double xSum2;
	double ySum2;
	double result;

	xMean = getDataMean(newData);
	yMean = getDataMean(oldData);

	sum1 = getDataSum1(xMean, yMean, newData, oldData);
	xSum2 = getDataSum2(newData, xMean );
	ySum2 = getDataSum2(oldData, yMean );

	result = sum1 / (sqrt(xSum2 * ySum2));

	return  abs(result);

}
