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

// 缓冲区定位枚举类型
typedef enum
{
    // 输入层 x1
    x1 = 0x00,
    // 输入层 x2
    x2,
    // 输入层 x3
    x3,
    // 隐藏层 h1
    h1,
    // 隐藏层 h2
    h2,
    // 隐藏层 h3
    h3,
    // 隐藏层 h4
    h4,
    // 输出层 o1
    o1,
    // 输出层 o2
    o2,
    // Loss 对 h1 的偏导数
    LTh1,
    // Loss 对 h2 的偏导数
    LTh2,
    // Loss 对 h3 的偏导数
    LTh3,
    // Loss 对 h4 的偏导数
    LTh4
} ucType;

// 模型结构体
typedef struct
{
    // 损失集（循环队列）
    double *loss, best_loss;
    unsigned char bare_rate;
    // 参数阵列
    double *W1, *W2, *B1, *B2;
    // 数据缓冲区(8*sizeof(double)字节大小)
    double *buffer;
    // 队列尾指针
    double *loss_end;
} xModel;

// 初始化 xModel
void pvInit(xModel **model);
// 销毁 xModel
void pvDeInit(xModel **model);
// 写入缓冲区
void pvWriteBuffer(xModel **model, ucType pos, double data);
// 读取缓冲区
void pvReadBuffer(xModel **model, ucType pos, double *data);
// 清除缓冲区
void pvClearBuffer(xModel **model);
// 前向传播
void pvForward(xModel **model, double *x, unsigned ulCount);
// 反向传播
double pvBackward(xModel **model, double lr, double *y, unsigned ulCount);
// 输出结果
int ulGetResultIndex(xModel **model);
// 提前终止训练
unsigned char pvEarlyStopDetect(xModel **model);
// 报告参数和损失
void pvDisplayWeights(xModel **model);
// 学习率衰减
double lr_fall(unsigned epoch);
// 无穷大标识符
extern const double inf;

#endif