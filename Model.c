#include "Model.h"

// 代表无穷大
const double inf = __DBL_MAX__;

// 容忍率
#define LOSS_NUM 50
// 学习率
#define LR 0.001
// 衰减率
#define GR 0.999
// 衰减保护轮数
#define GAMA_STEP 5000
// 迭代最大轮数
#define EPOCH 10000

// 初始化 xModel
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
    // 伪随机数 - 完全可复现，每次结果都是一样的
    double GaussData[26] =
        /* 高斯分布 N(0, 0.2) */
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

// 销毁 xModel
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

// 前向传播
void pvForward(xModel **model, double *_x, unsigned ulCount)
{
    xModel *pModel = *model;

    double sum = 0;
    for (int i = 0; i < 3; ++i)
        sum += _x[i];

    // 归一化层、隐藏层和输出层
    double *x = (double *)malloc(3 * sizeof(double));
    double *h = (double *)malloc(4 * sizeof(double));
    double *o = (double *)malloc(2 * sizeof(double));

    // 进行归一化
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

// 反向传播(返回这一个样本的 loss)
// 反向传播(返回这一个样本的 loss)
double pvBackward(xModel **model, double lr, double *y, unsigned ulCount)
{
    // 参数导数存储缓存
    double *dtBuf = (double *)malloc(26 * sizeof(double));
    // dL/dhi - 因为偏导符号似乎容易出现乱码所以用 dx 来表示了
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

    // 计算 Loss 作为提前停止训练凭证
    newLoss = CrossEntropy(o, y, 2);

    // dL/dh1 和 dL/dh2
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

    // 更新所有参数
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

// 输出结果
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

// 提前终止训练
unsigned char pvEarlyStopDetect(xModel **model)
{
    // 连续 bare_rate 次不下降就停止
    xModel *pTemp = *model;

    /* 这个数组实际上是在 DEBUG 的时候看到所有 loss 值 */
    // double lossSet[LOSS_NUM];
    // for (unsigned i = 0; i < pTemp->bare_rate; ++i)
    // {
    //     lossSet[i] = pTemp->loss[i];
    // }

    // 假设有一个loss，则 loss_end = loss + 1，前一个数据是新的
    unsigned i = 0;
    for (; i < pTemp->bare_rate; ++i)
    {
        // 如果最低的 loss 还在循环队列内说明不足五次
        if (pTemp->best_loss == pTemp->loss[i])
        {
            return (unsigned char)0;
        }
    }
    // 如果已经不在了，那么就一定超过了 LOSS_NUM 次
    return (unsigned char)1;
}

// 写入缓冲区
void pvWriteBuffer(xModel **model, ucType pos, double data)
{
    // uint32_t* 和 uint8_t 会合成 uint32_t*
    *((*model)->buffer + pos) = data;
}

// 读取缓冲区
void pvReadBuffer(xModel **model, ucType pos, double *data)
{
    // uint32_t* 和 uint8_t 会合成 uint32_t*
    *data = *((*model)->buffer + pos);
}

// 清除缓冲区
void pvClearBuffer(xModel **model)
{
    xModel *pTemp = *model;
    unsigned i = 0;
    for (; i < 8; ++i)
        *(pTemp->buffer + i) = 0;
}

// 学习率衰减
double lr_fall(unsigned epoch)
{
    if (epoch > GAMA_STEP)
        return pow(GR, epoch - GAMA_STEP);
    else
        return 1;
}

// 报告参数和损失
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
