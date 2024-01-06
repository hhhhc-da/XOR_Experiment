#include "Model.h"

// ���������
const double inf = __DBL_MAX__;

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
    // α����� - ��ȫ�ɸ��֣�ÿ�ν������һ����
    double GaussData[26] =
        /* ��˹�ֲ� N(0, 0.2) */
        {-1.55287, 0.343083, -2.12173, 1.28434, -1.24354, 0.557281, 1.16611, -3.67265, -0.571251, 1.96419, -1.99035, -3.21694, 0.0261942, -0.508873, 2.59592, -0.578999, 1.99263, 0.694865, -1.72447, 2.85001, 0.090228, 0.0137195, 0.123456, 0.0131554, 0.245881, 0.160267};

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

    printf("P(O) = [ %.6f, %.6f ]\n", o[0], o[1]);

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
