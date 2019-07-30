#ifndef ALGORITHMLIB_H
#define ALGORITHMLIB_H


#include <string>
#include <vector>
#include "svm.h"

using namespace std;


class libsvm
{

private:

    struct svm_model*		model_;
    struct svm_parameter	param;
    struct svm_problem		prob;
    struct svm_node*		x_space;
    double	normalData[256];

    //描述：释放模型内存（train、load_model内置调用）
    void free_model();

public:

	libsvm();
    ~libsvm();


    //描述： 建模、测试中的数据处理函数(内部调用)
    void noemal(int* basicData, int* backgroundData, int normalLength,double* normalData, string light);


    //描述： 将荧光建模数据进行数据处理
    //参数： int train[][256]                    采集建模数据的二维数组
    //      int trainLength                     二维数据的一维长度（每一类中的数据量）
    //      int exposeTimeData[256]             背景数据（荧光背景为黑暗环境下的测量数据，吸收光背景为白纸的测量数据)
    //      int startIndex                      有效数据起始位(256个数据中有效范围的起始位下标)
    //      int endIndex                        有效数据结束位(256个数据中有效范围的结束位下标,有效范围一般默认0~255）
    //      vector<vector<double>>& train_X     训练样本集(建模数组通过数据处理之后整合的数据集)
    //      int num                             样品类别标签（用于区分样品类别，一般为数字1,2,3...测量结果返回的也是该标签）
    //      vector<double>& train_Y             样本类别集(整合标签数据的数据集)
    void fluorescenceTrainHandl(int train[][256], int trainLength, int exposeTimeData[256], int startIndex, int endIndex, vector<vector<double>>& train_X, int num, vector<double>& train_Y);


    //描述： 将荧光测量数据进行数据处理
    //参数： int test[256]                       测量时采集的数据
    //      int exposeTimeData[256]             选中模型的背景数据
    //      int startIndex                      有效数据起始位(与鉴别模型的有效数据范围一致)
    //      int endIndex                        有效数据结束位(与鉴别模型的有效数据范围一致)
    //      vector<double>& test_X              测量数据通过数据处理之后整合的数据集
    void fluorescenceTestHandl(int test[256], int exposeTimeData[256], int startIndex, int endIndex, vector<double>& test_X);


    //描述： 将吸收光建模数据进行数据处理
    //参数： int train[][256]                    采集建模数据的二维数组
    //      int trainLength                     二维数据的一维长度（每一类中的数据量）
    //      int exposeTimeData[256]             背景数据（荧光背景为黑暗环境下的测量数据，吸收光背景为白纸的测量数据)
    //      int startIndex                      有效数据起始位(256个数据中有效范围的起始位下标)
    //      int endIndex                        有效数据结束位(256个数据中有效范围的结束位下标,有效范围一般默认0~255）
    //      vector<vector<double>>& train_X     训练样本集(建模数组通过数据处理之后整合的数据集)
    //      int num                             样品类别标签（用于区分样品类别，一般为数字1,2,3...测量结果返回的也是该标签）
    //      vector<double>& train_Y             样本类别集(整合标签数据的数据集)
    void absorbinglightTrainHandl(int train[][256], int trainLength, int exposeTimeData[256], int startIndex, int endIndex, vector<vector<double>>& train_X, int num, vector<double>& train_Y);


    //描述： 将吸收光测量数据进行数据处理
    //参数： int test[256]                       测量时采集的数据
    //      int exposeTimeData[256]             选中模型的背景数据
    //      int startIndex                      有效数据起始位(与鉴别模型的有效数据范围一致)
    //      int endIndex                        有效数据结束位(与鉴别模型的有效数据范围一致)
    //      vector<double>& test_X              测量数据通过数据处理之后整合的数据集
    void absorbinglightTestHandl(int test[256], int exposeTimeData[256], int startIndex, int endIndex, vector<double>& test_X);


    //描述：对采集数据进行训练，并保存模型文件（txt）
    //参数：const vector<vector<double>>& x    数据集
    //     const vector<double>& y            训练样本集
    //     string model_path                  模型路径名称
    void train(const vector<vector<double>>& x, const vector<double>& y, string model_path);


    //描 述：  测量数据集对比模型，判别样品
    //参 数：  string model_path             数据集
    //        const vector<double>& y       训练样本集
    //返回值：  double  value                预测样品类别标签
    int predict(string model_path, const vector<double>& x);


    //描述：导入模型（ predict内置调用 ）
    int load_model(string model_path);


    //描述：保存模型（ train内置调用 ）
    int save_model(string model_path);


    //描述：数据处理（pearsonCorrelation内置调用）
    double  getDataMean(int* newData);
    double	getDataSum1(double xMean, double yMean, int* xData, int* yData);
    double	getDataSum2(int* Data, double Mean);


    //描  述：   判断测量数据与判别结果数据的相似度
    //参  数：   int* newData      测量数据
    //          int* oldData      判别结果所属类别中的一组数据
    //返回值：    double ps         相似度（判定小于0.90为失败）
    double	pearsonCorrelation(int* newData, int* oldData);

};

#endif // ALGORITHMLIB_H
