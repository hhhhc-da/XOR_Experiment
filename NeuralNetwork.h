#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

// ReLU 线性修正函数
double ReLU(double dMem);
// Logistic 函数
void Logistic(double **pMem, unsigned ulCount);
// 交叉熵 cross enrtopy 函数
double CrossEntropy(double *pMem, double *pLabel, unsigned ulCount);

#endif