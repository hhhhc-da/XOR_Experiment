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