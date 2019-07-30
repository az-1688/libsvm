#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include "algorithm.h"
#include "svm.h"

using namespace std;
//内存分配
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* 封装svm                                                                     */
/************************************************************************/

//************************************
// 描    述: 构造函数
// 方    法: CxLibSVM
// 文 件 名: CxLibSVM::CxLibSVM
// 访问权限: public 
// 返 回 值: 
// 限 定 符:
//************************************
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include "algorithm.h"
#include "svm.h"

using namespace std;
//内存分配
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* 封装svm                                                                     */
/************************************************************************/


libsvm::libsvm()
{
	model_ = NULL;
}


libsvm::~libsvm()
{
	free_model();
}

//数据归一化函数
void	libsvm::noemal(int* basicData, int* backgroundData, int normalLength, double* normalData, string light)
{
	double data[256];
	double minValue;
	double maxValue;
	//double normalValue;

	if (light == "absorbinglight")
	{
		//printf("吸收光");

		//获取相除数据
		for (int i = 0; i < normalLength; i++)
		{
			data[i] = basicData[i] / backgroundData[i];
		}
	}
	else if (light == "fluorescence")
	{
		//printf("荧光");

		//获取扣减数据
		for (int i = 0; i < normalLength; i++)
		{
			data[i] = basicData[i] - backgroundData[i];
		}
	}


	//获取最大最小值
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

	//进行归一化
	for (int i = 0; i < normalLength; i++)
	{
		normalData[i] = double((data[i] - minValue)) / double((maxValue - minValue));
	}
}




// 荧光建模数据处理(可选择数据范围)
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





// 荧光测量数据处理
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



// 吸收光建模数据处理
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



// 吸收光测量数据处理
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

	//释放先前的模型
	free_model();

	/*初始化*/
	long	len = x.size();
	long	dim = x[0].size();
	long	elements = len * dim;

	//参数初始化，参数调整部分在这里修改即可
	// 默认参数
	param.svm_type = C_SVC;		//算法类型
	param.kernel_type = LINEAR;	//核函数类型
	param.degree = 3;	//多项式核函数的参数degree
	param.coef0 = 0;	//多项式核函数的参数coef0
	param.gamma = 0.5;	//1/num_features，rbf核函数参数
	param.nu = 0.5;		//nu-svc的参数
	param.C = 10;		//正则项的惩罚系数
	param.eps = 1e-3;	//收敛精度
	param.cache_size = 100;	//求解的内存缓冲 100MB
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;	//1表示训练时生成概率模型，0表示训练时不生成概率模型，用于预测样本的所属类别的概率
	param.nr_weight = 0;	//类别权重
	param.weight = NULL;	//样本权重
	param.weight_label = NULL;	//类别权重


	//转换数据为libsvm格式
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

	/*训练*/
	model_ = svm_train(&prob, &param);

	cout << "********************" << endl;
	cout << model_ << endl;
	cout << "********************" << endl;

	save_model(model_path);			// 保存模型
}




	//************************************
	// 描    述: 预测测试样本所属类别和概率
	// 方    法: predict
	// 文 件 名: CxLibSVM::predict
	// 访问权限: public 
	// 参    数: const vector<double> & x	样本
	// 参    数: double & prob_est			类别估计的概率
	// 返 回 值: double						预测的类别
	// 限 定 符:
	//************************************
	//int predict(const vector<double>& x,double& prob_est)
	//{
	//	//数据转换
	//	svm_node* x_test = Malloc(struct svm_node, x.size()+1);
	//	for (unsigned int i=0; i<x.size(); i++)
	//	{
	//		x_test[i].index = i;
	//		x_test[i].value = x[i];
	//	}
	//	x_test[x.size()].index = -1;
	//	double *probs = new double[model_->nr_class];//存储了所有类别的概率
	//	//预测类别和概率
	//	int value = (int)svm_predict_probability(model_, x_test, probs);
	//	for (int k = 0; k < model_->nr_class;k++)
	//	{//查找类别相对应的概率
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
	load_model(model_path);		// 导入模型

	//数据转换
	svm_node* x_test = Malloc(struct svm_node, x.size() + 1);
	for (unsigned int i = 0; i < x.size(); i++)
	{
		x_test[i].index = i;
		x_test[i].value = x[i];
	}
	x_test[x.size()].index = -1;
	double* probs = new double[model_->nr_class];//存储了所有类别的概率

	//预测类别和概率
	int value = (int)svm_predict_probability(model_, x_test, probs);

	delete[] probs;
	return value;
}




	//************************************
	// 描    述: 导入svm模型
	// 方    法: load_model
	// 访问权限: public 
	// 参    数: string model_path	模型路径
	// 返 回 值: int 0表示成功；-1表示失败
	// 限 定 符:
	//************************************
int libsvm::load_model(string model_path)
{
	//释放原来的模型
	free_model();
	//导入模型
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

//皮尔森系数判别（两条原始数据）
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
