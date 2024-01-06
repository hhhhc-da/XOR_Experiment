#include "NeuralNetwork.h"

double ReLU(double dMem)
{
    if (dMem > 0)
        return dMem;
    else
        return 0;
}

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

double CrossEntropy(double *pMem, double *pLabel, unsigned ulCount)
{
    double ret = 0;

    unsigned i = 0;
    for (; i < ulCount; ++i)
        ret += (*(pLabel + i)) * log(*(pMem + i));

    return -ret / ulCount;
}

// int main(void)
// {
//     double x = 2, y = -2;

//     printf("x: %f, y: %f\n", x, y);
//     printf("relu_x: %f, relu_y: %f\n", ReLU(x), ReLU(y));

//     double *data = (double *)malloc(2 * sizeof(double));
//     *data = 2;
//     *(data + 1) = 4;

//     printf("\nx: %f, y: %f\n", *data, *(data + 1));
//     Logistic(&data, 2);
//     printf("logistic_x: %f, logistic_y: %f\n", *data, *(data + 1));

//     double label[2] = {0, 1};
//     double loss = CrossEntropy(data, label, 2);

//     printf("\nlabel_x: %f, label_y: %f\n", *label, *(label + 1));
//     printf("loss: %f", loss);

//     free(data);

//     return 0;
// }
