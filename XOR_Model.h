#ifndef XOR_MODEL_H
#define XOR_MODEL_H

#include "NeuralNetwork.h"
#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

// ��������λö������
typedef enum
{
    // ����� x1
    x1 = 0x00,
    // ����� x2
    x2,
    // ���ز� h1
    h1,
    // ���ز� h2
    h2,
    // ����� o1
    o1,
    // ����� o2
    o2,
    // Loss �� h1 ��ƫ����
    LTh1,
    // Loss �� h2 ��ƫ����
    LTh2
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

/* ��˹�ֲ� N(0, 0.2)��������ʼ��ֵ���� */
// 1.39955, 0.789535, 2.875, -2.74442, -1.51085, 0.367908, 1.45579, 1.19469, -2.13551, -0.0324631, 2.213, 3.40381
//    b1       b1'      b2       b2'      w11       w11'     w12      w12'      w21        w21'     w22     w22'

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
// ������ʶ��
extern const double inf;

#endif