#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
using namespace std;

#define GAUSS_NUM 26

string tmp = "";
unsigned index = 0;
double GaussData[GAUSS_NUM];
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<double> dist(0, 2.3);
std::uniform_int_distribution<int> rd_index(0, 7);

// ������
#define LOSS_NUM 50
// ѧϰ��
#define LR 0.001
// ˥����
#define GR 0.999
// ˥����������
#define GAMA_STEP 5000
// �����������
#define EPOCH 10000
// Ѱ���������
#define FIND_EPOCH 500000

// ���������
const double inf = __DBL_MAX__;

// ��������λö������
typedef enum
{
    // ����� x1
    x1 = 0x00,
    // ����� x2
    x2,
    // ����� x3
    x3,
    // ���ز� h1
    h1,
    // ���ز� h2
    h2,
    // ���ز� h3
    h3,
    // ���ز� h4
    h4,
    // ����� o1
    o1,
    // ����� o2
    o2,
    // Loss �� h1 ��ƫ����
    LTh1,
    // Loss �� h2 ��ƫ����
    LTh2,
    // Loss �� h3 ��ƫ����
    LTh3,
    // Loss �� h4 ��ƫ����
    LTh4
} ucType;

// ģ�ͽṹ��
typedef struct
{
    // ��ʧ����ѭ�����У�
    double *loss, best_loss;
    unsigned char bare_rate;
    // ��������
    double *W1, *W2, *B1, *B2;
    // ���ݻ�����(8*sizeof(double)�ֽڴ�С)
    double *buffer;
    // ����βָ��
    double *loss_end;
} xModel;

// ReLU ������������
double ReLU(double dMem);
// Logistic ����
void Logistic(double **pMem, unsigned ulCount);
// ������ cross enrtopy ����
double CrossEntropy(double *pMem, double *pLabel, unsigned ulCount);

// ��ʼ�� xModel
void pvInit(xModel **model);
// ���� xModel
void pvDeInit(xModel **model);
// д�뻺����
void pvWriteBuffer(xModel **model, ucType pos, double data);
// ��ȡ������
void pvReadBuffer(xModel **model, ucType pos, double *data);
// ���������
void pvClearBuffer(xModel **model);
// ǰ�򴫲�
void pvForward(xModel **model, double *x, unsigned ulCount);
// ���򴫲�
double pvBackward(xModel **model, double lr, double *y, unsigned ulCount);
// ������
int ulGetResultIndex(xModel **model);
// ��ǰ��ֹѵ��
unsigned char pvEarlyStopDetect(xModel **model);
// �����������ʧ
void pvDisplayWeights(xModel **model);
// ѧϰ��˥��
double lr_fall(unsigned epoch);

int main(void)
{
    auto time_start = std::chrono::steady_clock::now();
    // ѵ��������ʽ���� Dark_Function(input[3k],input[3k+1],input[3k+2]) = output[2k]
    double input[24] = {0, 0, 0,
                        0, 0, 1,
                        0, 1, 0,
                        0, 1, 1,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        1, 1, 1};
    // ͬʱ���ǵ�ѵ������Ҫ���й�һ��
    /* �� Xi = Xi/��(X��i) , ���� Xi �� X ��ѵ����������Ԫ��, X��i �������е�ȫ��Ԫ�� */

    // �¹�������
    /* f(0, 0, 0) = 0  f(0, 0, 1) = 1
     * f(0, 1, 0) = 1  f(0, 1, 1) = 0
     * f(1, 0, 0) = 1  f(1, 0, 1) = 0
     * f(1, 1, 0) = 0  f(1, 1, 1) = 1
     */
    double output[16] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};

    unsigned find_count = 0;
    while (find_count++ < FIND_EPOCH)
    {
        if (find_count % 20 == 0)
            printf("Find Count - [%u]\n", find_count);

        xModel *pModel = (xModel *)malloc(sizeof(xModel));
        pvInit(&pModel);

        unsigned count = 0;

        // do
        // {
        //     // if (count % 20 == 0)
        //     //     printf("\n\nEpoch[%u]\n", ++count);

        //     unsigned i = 0;
        //     double loss = 0;

        //     // ���������α�����, 20 �γ���ȡ��ֵ
        //     for (; i < 20; ++i)
        //     {
        //         unsigned rdi = rd_index(gen);
        //         pvForward(&pModel, &input[3 * rdi], 3);
        //         loss += pvBackward(&pModel, LR * lr_fall(count), &output[2 * rdi], 2);
        //     }

        //     if (pModel->loss_end - pModel->loss == LOSS_NUM - 1)
        //     {
        //         *(pModel->loss_end) = loss;
        //         pModel->loss_end = pModel->loss;
        //     }
        //     else
        //     {
        //         *(pModel->loss_end) = loss;
        //         ++(pModel->loss_end);
        //     }
        //     // ������� loss
        //     if (loss < pModel->best_loss)
        //     {
        //         pModel->best_loss = loss;
        //     }
        //     // pvDisplayWeights(&pModel);

        // } while (!pvEarlyStopDetect(&pModel) && count < EPOCH);

        int ret[8];

        for (int i = 0; i < 8; ++i)
        {
            pvForward(&pModel, &input[3 * i], 3);
            ret[i] = ulGetResultIndex(&pModel);
        }

        unsigned char owari = 1;

        for (int i = 0; i < 8; ++i)
        {
            if (ret[i] != output[2 * i + 1])
            {
                owari = 0;
                break;
            }
        }

        if (owari == 1)
        {
            printf("\t���ɹ�: return ");
            for (int i = 0; i < 8; ++i)
                printf("%u ", ret[i]);
            printf("\n\n");

            // cout << "{";
            // for (int i = 0; i < GAUSS_NUM - 1; ++i)
            // {
            //     cout << GaussData[i] << ",";
            // }
            // cout << GaussData[GAUSS_NUM - 1] << "}";

            pvDisplayWeights(&pModel);

            ofstream of("D:\\pandownload1\\Desktop\\CodingFolder\\C\\Encoder\\XOR_Debug\\result.txt", ios::trunc);

            if (!of.good())
            {
                cout << "���ļ� result.txt ʧ��" << endl;
            }
            else
            {
                of << "{";
                for (int i = 0; i < GAUSS_NUM - 1; ++i)
                    of << GaussData[i] << ",";
                of << GaussData[GAUSS_NUM - 1] << "}\n";

                of << "{";
                for (int i = 0; i < 12; ++i)
                    of << pModel->W1[i] << ",";
                for (int i = 0; i < 8; ++i)
                    of << pModel->W2[i] << ",";
                for (int i = 0; i < 4; ++i)
                    of << pModel->B1[i] << ",";

                of << pModel->B2[0] << "," << pModel->B2[1] << "}" << flush;
                of.close();

                pvDeInit(&pModel);
                free(pModel);
                break;
            }
        }
        else
        {
            // printf("\t���ʧ��: return ");
            // for (int i = 0; i < 8; ++i)
            //     printf("%u ", ret[i]);
            // printf("\n");
        }

        pvDeInit(&pModel);
        free(pModel);
    }

    auto time_end = std::chrono::steady_clock::now();
    double dr_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();

    cout << "\n����ʱ��: " << dr_ms / 1000.0 << " s" << endl;

    system("pause");
    return 0;
}

// ��ʼ�� xModel
void pvInit(xModel **model)
{
    xModel *pTemp = *model;

    pTemp->W1 = (double *)malloc(12 * sizeof(double));
    pTemp->W2 = (double *)malloc(8 * sizeof(double));
    pTemp->B1 = (double *)malloc(4 * sizeof(double));
    pTemp->B2 = (double *)malloc(2 * sizeof(double));

    pTemp->buffer = (double *)malloc(13 * sizeof(double));
    pTemp->bare_rate = LOSS_NUM;

    pTemp->loss = (double *)malloc(pTemp->bare_rate * sizeof(double));
    pTemp->loss_end = pTemp->loss;

    pTemp->best_loss = inf;

    unsigned i = 0, count = 0;
    // // α������������˹�ֲ�
    // index = 0;
    // for (unsigned j = 0; j < GAUSS_NUM; ++j)
    // {
    //     GaussData[j] = dist(gen);
    // }

    double GaussData[26] =
        /* ��˹�ֲ� N(0, 0.2) */
        {-1.55287, 0.408441, -2.12173, 1.27837, -1.24401, 0.583065, 1.2051, -3.67265, -0.523984, 1.97568, -1.99259, -3.21694, 0.0261942, -0.473885, 2.60909, -0.578997, 2.00597, 0.729969, -1.71852, 2.85951, 0.0653581, 0.0193447, 0.0862588, 0.00925318, 0.172263, 0.114868};

    for (i = 0; i < 12; ++i)
    {
        pTemp->W1[i] = GaussData[count++];
    }

    for (i = 0; i < 8; ++i)
    {
        pTemp->W2[i] = GaussData[count++];
    }

    for (i = 0; i < 4; ++i)
    {
        pTemp->B1[i] = GaussData[count++];
    }

    for (i = 0; i < 2; ++i)
    {
        pTemp->B2[i] = GaussData[count++];
    }

    for (i = 0; i < LOSS_NUM; ++i)
    {
        pTemp->loss[i] = inf;
    }
    for (i = 0; i < 13; ++i)
    {
        pTemp->buffer[i] = 0;
    }
}

// ���� xModel
void pvDeInit(xModel **model)
{
    xModel *pTemp = *model;
    free(pTemp->loss);
    free(pTemp->B1);
    free(pTemp->B2);
    free(pTemp->W1);
    free(pTemp->W2);
    free(pTemp->buffer);
}

// ǰ�򴫲�
void pvForward(xModel **model, double *_x, unsigned ulCount)
{
    xModel *pModel = *model;

    double sum = 0;
    for (int i = 0; i < 3; ++i)
        sum += _x[i];

    // ��һ���㡢���ز�������
    double *x = (double *)malloc(3 * sizeof(double));
    double *h = (double *)malloc(4 * sizeof(double));
    double *o = (double *)malloc(2 * sizeof(double));

    // ���й�һ��
    for (int i = 0; i < 3; ++i)
        x[i] = _x[i] / sum;

    pvWriteBuffer(&pModel, x1, x[0]);
    pvWriteBuffer(&pModel, x2, x[1]);
    pvWriteBuffer(&pModel, x3, x[2]);

    h[0] = ReLU(pModel->W1[0] * x[0] + pModel->W1[1] * x[1] + pModel->W1[2] * x[2] + pModel->B1[0]);
    h[1] = ReLU(pModel->W1[3] * x[0] + pModel->W1[4] * x[1] + pModel->W1[5] * x[2] + pModel->B1[1]);
    h[2] = ReLU(pModel->W1[6] * x[0] + pModel->W1[7] * x[1] + pModel->W1[8] * x[2] + pModel->B1[2]);
    h[3] = ReLU(pModel->W1[9] * x[0] + pModel->W1[10] * x[1] + pModel->W1[11] * x[2] + pModel->B1[3]);

    pvWriteBuffer(&pModel, h1, h[0]);
    pvWriteBuffer(&pModel, h2, h[1]);
    pvWriteBuffer(&pModel, h3, h[2]);
    pvWriteBuffer(&pModel, h4, h[3]);

    o[0] = pModel->W2[0] * h[0] + pModel->W2[1] * h[1] + pModel->W2[2] * h[2] + pModel->W2[3] * h[3] + pModel->B2[0];
    o[1] = pModel->W2[4] * h[0] + pModel->W2[5] * h[1] + pModel->W2[6] * h[2] + pModel->W2[7] * h[3] + pModel->B2[1];

    Logistic(&o, 2);

    pvWriteBuffer(&pModel, o1, o[0]);
    pvWriteBuffer(&pModel, o2, o[1]);

    free(x);
    free(h);
    free(o);
}

// ���򴫲�(������һ�������� loss)
double pvBackward(xModel **model, double lr, double *y, unsigned ulCount)
{
    // ���������洢����
    double *dtBuf = (double *)malloc(26 * sizeof(double));
    // dL/dhi - ��Ϊƫ�������ƺ����׳������������� dx ����ʾ��
    double *dtMid = (double *)malloc(4 * sizeof(double));
    xModel *pModel = *model;

    double newLoss = inf;

    double o[2] = {-1, -1};
    pvReadBuffer(&pModel, o1, &o[0]);
    pvReadBuffer(&pModel, o2, &o[1]);

    double h[4] = {-1, -1, -1, -1};
    pvReadBuffer(&pModel, h1, &h[0]);
    pvReadBuffer(&pModel, h2, &h[1]);
    pvReadBuffer(&pModel, h3, &h[2]);
    pvReadBuffer(&pModel, h4, &h[3]);

    double x[3] = {-1, -1, -1};
    pvReadBuffer(&pModel, x1, &x[0]);
    pvReadBuffer(&pModel, x2, &x[1]);
    pvReadBuffer(&pModel, x3, &x[2]);

    // ���� Loss ��Ϊ��ǰֹͣѵ��ƾ֤
    newLoss = CrossEntropy(o, y, 2);

    // dL/dh1 �� dL/dh2
    for (int i = 0; i < 4; ++i)
        dtMid[i] = pModel->W2[i] * (1 - o[0]) * (-y[0] / 2) + pModel->W2[4 + i] * (1 - o[1]) * (-y[1] / 2);

    // dL/dW2, dL/dB2
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
            dtBuf[12 + j + 4 * i] = h[j] * (1 - o[i]) * (-y[i] / 2);

        dtBuf[24 + i] = (1 - o[i]) * (-y[i] / 2);
    }

    // dL/dW1, dL/dB1
    for (int i = 0; i < 4; ++i)
    {
        if (h[i] > 0)
        {
            for (int j = 0; j < 3; ++j)
                dtBuf[j + 3 * i] = x[j] * dtMid[i];

            dtBuf[20 + i] = dtMid[i];
        }
        else
        {
            for (int j = 0; j < 3; ++j)
                dtBuf[j + 3 * i] = 0;

            dtBuf[20 + i] = 0;
        }
    }

    // �������в���
    // W1 = W1 - dL/dW1
    for (int i = 0; i < 12; ++i)
        pModel->W1[i] -= dtBuf[i] * lr;
    // W2 = W2 - dL/dW2
    for (int i = 0; i < 8; ++i)
        pModel->W2[i] -= dtBuf[12 + i] * lr;
    // B1 = B1 - dL/dB1
    for (int i = 0; i < 4; ++i)
        pModel->B1[i] -= dtBuf[20 + i] * lr;
    // B2 = B2 - dL/dB2
    for (int i = 0; i < 2; ++i)
        pModel->B2[i] -= dtBuf[24 + i] * lr;

    free(dtBuf);
    free(dtMid);

    return newLoss;
}

// ������
int ulGetResultIndex(xModel **model)
{
    xModel *pTemp = *model;

    double o[2] = {-1, -1};
    pvReadBuffer(&pTemp, o1, &o[0]);
    pvReadBuffer(&pTemp, o2, &o[1]);

    // printf("p(o) = [ %.6f, %.6f ]\n", o[0], o[1]);

    if (o[0] != -1 && o[1] != -1 && o[0] >= o[1])
        return 0;
    else if (o[0] != -1 && o[1] != -1 && o[0] < o[1])
        return 1;
    else
        return -1;
}

// ��ǰ��ֹѵ��
unsigned char pvEarlyStopDetect(xModel **model)
{
    // ���� bare_rate �β��½���ֹͣ
    xModel *pTemp = *model;

    /* �������ʵ�������� DEBUG ��ʱ�򿴵����� loss ֵ */
    // double lossSet[LOSS_NUM];
    // for (unsigned i = 0; i < pTemp->bare_rate; ++i)
    // {
    //     lossSet[i] = pTemp->loss[i];
    // }

    // ������һ��loss���� loss_end = loss + 1��ǰһ���������µ�
    unsigned i = 0;
    for (; i < pTemp->bare_rate; ++i)
    {
        // �����͵� loss ����ѭ��������˵���������
        if (pTemp->best_loss == pTemp->loss[i])
        {
            return (unsigned char)0;
        }
    }
    // ����Ѿ������ˣ���ô��һ�������� LOSS_NUM ��
    return (unsigned char)1;
}

// д�뻺����
void pvWriteBuffer(xModel **model, ucType pos, double data)
{
    // uint32_t* �� uint8_t ��ϳ� uint32_t*
    *((*model)->buffer + pos) = data;
}

// ��ȡ������
void pvReadBuffer(xModel **model, ucType pos, double *data)
{
    // uint32_t* �� uint8_t ��ϳ� uint32_t*
    *data = *((*model)->buffer + pos);
}

// ���������
void pvClearBuffer(xModel **model)
{
    xModel *pTemp = *model;
    unsigned i = 0;
    for (; i < 8; ++i)
        *(pTemp->buffer + i) = 0;
}

// ѧϰ��˥��
double lr_fall(unsigned epoch)
{
    if (epoch > GAMA_STEP)
        return pow(GR, epoch - GAMA_STEP);
    else
        return 1;
}

// �����������ʧ
void pvDisplayWeights(xModel **model)
{
    xModel *pModel = *model;
    double newLoss = inf;
    if (pModel->loss_end - pModel->loss == 0)
    {
        newLoss = *(pModel->loss + LOSS_NUM - 1);
    }
    else
    {
        newLoss = *(pModel->loss_end - 1);
    }

    printf("\n");
    printf("Loss: %.6f\n\n", newLoss);
    printf("W:| %03.06f %03.06f %03.06f | B:| %03.06f |\n", pModel->W1[0], pModel->W1[1], pModel->W1[2], pModel->B1[0]);
    printf("  | %03.06f %03.06f %03.06f |   | %03.06f |\n", pModel->W1[3], pModel->W1[4], pModel->W1[5], pModel->B1[1]);
    printf("  | %03.06f %03.06f %03.06f |   | %03.06f |\n", pModel->W1[6], pModel->W1[7], pModel->W1[8], pModel->B1[2]);
    printf("  | %03.06f %03.06f %03.06f |   | %03.06f |\n\n", pModel->W1[9], pModel->W1[10], pModel->W1[11], pModel->B1[3]);
    printf("W':| %03.06f %03.06f %03.06f %03.06f | B':| %03.06f |\n", pModel->W2[0], pModel->W2[1], pModel->W2[2], pModel->W2[3], pModel->B2[0]);
    printf("   | %03.06f %03.06f %03.06f %03.06f |    | %03.06f |\n\n", pModel->W2[4], pModel->W2[5], pModel->W2[6], pModel->W2[7], pModel->B2[1]);
}

// ReLU ����
double ReLU(double dMem)
{
    if (dMem > 0)
        return dMem;
    else
        return 0;
}

// Logistic ������
void Logistic(double **pMem, unsigned ulCount)
{
    double *pMemory = (double *)malloc(ulCount * sizeof(double));
    double *pTmp = *pMem, *pTmp2 = pMemory;
    double total = 0;

    unsigned i = 0;
    for (; i < ulCount; ++i)
        total += exp(*pTmp++);

    pTmp = *pMem;
    for (i = 0; i < ulCount; ++i)
        *pTmp2++ = exp(*(pTmp + i)) / total;

    free(*pMem);
    *pMem = pMemory;
}

// ��������ʧ����
double CrossEntropy(double *pMem, double *pLabel, unsigned ulCount)
{
    double ret = 0;

    unsigned i = 0;
    for (; i < ulCount; ++i)
        ret += (*(pLabel + i)) * log(*(pMem + i));

    return -ret / ulCount;
}