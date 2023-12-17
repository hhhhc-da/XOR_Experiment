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
    // 隐藏层 h1
    h1,
    // 隐藏层 h2
    h2,
    // 输出层 o1
    o1,
    // 输出层 o2
    o2,
    // Loss 对 h1 的偏导数
    LTh1,
    // Loss 对 h2 的偏导数
    LTh2
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

/* 高斯分布 N(0, 0.2)，参数初始化值如下 */
// 1.39955, 0.789535, 2.875, -2.74442, -1.51085, 0.367908, 1.45579, 1.19469, -2.13551, -0.0324631, 2.213, 3.40381
//    b1       b1'      b2       b2'      w11       w11'     w12      w12'      w21        w21'     w22     w22'

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