#include "XOR_Model.h"

// ���������
const double inf = __DBL_MAX__;
// ˥����������
#define GAMA_STEP 100
// ������
#define LOSS_NUM 10
// ˥����
#define GR 0.99

// ��ʼ�� xModel
void pvInit(xModel **model)
{
    xModel *pTemp = *model;

    pTemp->W1 = (double *)malloc(4 * sizeof(double));
    pTemp->W2 = (double *)malloc(4 * sizeof(double));
    pTemp->B1 = (double *)malloc(2 * sizeof(double));
    pTemp->B2 = (double *)malloc(2 * sizeof(double));

    pTemp->buffer = (double *)malloc(8 * sizeof(double));
    pTemp->bare_rate = LOSS_NUM;

    pTemp->loss = (double *)malloc(pTemp->bare_rate * sizeof(double));
    pTemp->loss_end = pTemp->loss;

    pTemp->best_loss = inf;

    unsigned i = 0, count = 0;
    // α����� - ��ȫ�ɸ��֣�ÿ�ν������һ����
    double GaussData[12] =
        /* ��˹�ֲ� N(0, 0.2) */
        {1.39955, 0.789535, 2.875, -2.74442, -1.51085, 0.367908, 1.45579, 1.19469, -2.13551, -0.0324631, 2.213, 3.40381};

    for (; i < 2; ++i)
    {
        pTemp->B1[i] = GaussData[count++];
        pTemp->B2[i] = GaussData[count++];
    }
    for (i = 0; i < 4; ++i)
    {
        pTemp->W1[i] = GaussData[count++];
        pTemp->W2[i] = GaussData[count++];
    }
    for (i = 0; i < 5; ++i)
    {
        pTemp->loss[i] = inf;
    }
    for (i = 0; i < 8; ++i)
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
void pvForward(xModel **model, double *x, unsigned ulCount)
{
    xModel *pModel = *model;
    // ���ز�������
    double *h = (double *)malloc(2 * sizeof(double));
    double *o = (double *)malloc(2 * sizeof(double));

    pvWriteBuffer(&pModel, x1, x[0]);
    pvWriteBuffer(&pModel, x2, x[1]);

    h[0] = ReLU(pModel->W1[0] * x[0] + pModel->W1[1] * x[1] + pModel->B1[0]);
    h[1] = ReLU(pModel->W1[2] * x[0] + pModel->W1[3] * x[1] + pModel->B1[1]);

    pvWriteBuffer(&pModel, h1, h[0]);
    pvWriteBuffer(&pModel, h2, h[1]);

    o[0] = pModel->W2[0] * h[0] + pModel->W2[1] * h[1] + pModel->B2[0];
    o[1] = pModel->W2[2] * h[0] + pModel->W2[3] * h[1] + pModel->B2[1];

    Logistic(&o, 2);

    pvWriteBuffer(&pModel, o1, o[0]);
    pvWriteBuffer(&pModel, o2, o[1]);

    free(h);
    free(o);
}

// ���򴫲�(������һ�������� loss)
double pvBackward(xModel **model, double lr, double *y, unsigned ulCount)
{
    // ���������洢����
    double *dtBuf = (double *)malloc(12 * sizeof(double));
    // dL/dh1 �� dL/dh2 - ��Ϊƫ�������ƺ����׳������������� dx ����ʾ��
    double *dtMid = (double *)malloc(2 * sizeof(double));
    xModel *pModel = *model;

    double newLoss = inf;

    double o[2] = {-1, -1};
    pvReadBuffer(&pModel, o1, &o[0]);
    pvReadBuffer(&pModel, o2, &o[1]);

    double h[2] = {-1, -1};
    pvReadBuffer(&pModel, h1, &h[0]);
    pvReadBuffer(&pModel, h2, &h[1]);

    double x[2] = {-1, -1};
    pvReadBuffer(&pModel, x1, &x[0]);
    pvReadBuffer(&pModel, x2, &x[1]);

    // ���� Loss ��Ϊ��ǰֹͣѵ��ƾ֤
    newLoss = CrossEntropy(o, y, 2);

    // dL/dh1 �� dL/dh2
    dtMid[0] = pModel->W2[0] * (1 - o[0]) * (-y[0] / 2) + pModel->W2[2] * (1 - o[1]) * (-y[1] / 2);
    dtMid[1] = pModel->W2[1] * (1 - o[0]) * (-y[0] / 2) + pModel->W2[3] * (1 - o[1]) * (-y[1] / 2);

    // dL/dW2
    dtBuf[0] = h[0] * (1 - o[0]) * (-y[0] / 2);
    dtBuf[1] = h[1] * (1 - o[0]) * (-y[0] / 2);
    dtBuf[2] = h[0] * (1 - o[1]) * (-y[1] / 2);
    dtBuf[3] = h[1] * (1 - o[1]) * (-y[1] / 2);

    // dL/dB2
    dtBuf[4] = (1 - o[0]) * (-y[0] / 2);
    dtBuf[5] = (1 - o[1]) * (-y[1] / 2);

    // dL/dW1, dL/dB1
    if (h[0] > 0)
    {
        dtBuf[6] = x[0] * dtMid[0];
        dtBuf[7] = x[1] * dtMid[0];
        dtBuf[10] = dtMid[0];
    }
    else
    {
        dtBuf[6] = 0;
        dtBuf[7] = 0;
        dtBuf[10] = 0;
    }

    if (h[1] > 0)
    {
        dtBuf[8] = x[0] * dtMid[1];
        dtBuf[9] = x[1] * dtMid[1];
        dtBuf[11] = dtMid[1];
    }
    else
    {
        dtBuf[8] = 0;
        dtBuf[9] = 0;
        dtBuf[11] = 0;
    }

    // �������в���
    // W2 = W2 - dL/dW2
    pModel->W2[0] -= dtBuf[0] * lr;
    pModel->W2[1] -= dtBuf[1] * lr;
    pModel->W2[2] -= dtBuf[2] * lr;
    pModel->W2[3] -= dtBuf[3] * lr;

    // B2 = B2 - dL/dB2
    pModel->B2[0] -= dtBuf[4] * lr;
    pModel->B2[1] -= dtBuf[5] * lr;

    // W1 = W1 - dL/dW1
    pModel->W1[0] -= dtBuf[6] * lr;
    pModel->W1[1] -= dtBuf[7] * lr;
    pModel->W1[2] -= dtBuf[8] * lr;
    pModel->W1[3] -= dtBuf[9] * lr;

    // B1 = B1 - dL/dB1
    pModel->B1[0] -= dtBuf[10] * lr;
    pModel->B1[1] -= dtBuf[11] * lr;

    return newLoss;
}

// ������
int ulGetResultIndex(xModel **model)
{
    xModel *pTemp = *model;

    double o[2] = {-1, -1};
    pvReadBuffer(&pTemp, o1, &o[0]);
    pvReadBuffer(&pTemp, o2, &o[1]);

    printf("p(o) = [ %.6f, %.6f ]\n", o[0], o[1]);

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
    // ����Ѿ������ˣ���ô��һ�����������
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
        newLoss = *(pModel->loss + 4);
    }
    else
    {
        newLoss = *(pModel->loss_end - 1);
    }

    printf("\n");
    printf("W:| %.6f %.6f | B:| %.6f | W':| %.6f %.6f | B':| %.6f | \tLoss: %.6f\n", pModel->W1[0], pModel->W1[1], pModel->B1[0], pModel->W2[0], pModel->W2[1], pModel->B2[0], newLoss);
    printf("  | %.6f %.6f |   | %.6f |    | %.6f %.6f |    | %.6f |\n", pModel->W1[2], pModel->W1[3], pModel->B1[1], pModel->W2[2], pModel->W2[3], pModel->B2[1]);
}
