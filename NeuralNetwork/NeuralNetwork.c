#include "NeuralNetwork.h"

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
