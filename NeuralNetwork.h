#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

// ReLU ������������
double ReLU(double dMem);
// Logistic ����
void Logistic(double **pMem, unsigned ulCount);
// ������ cross enrtopy ����
double CrossEntropy(double *pMem, double *pLabel, unsigned ulCount);

#endif