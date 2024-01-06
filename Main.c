#include "Model.h"

// 参数总个数
#define GAUSS_NUM 26
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

unsigned getRondomIndex(void)
{
    /* 调用 systick 获取硬件随机数什么的 */
    // 这里就使用可以在 PC 运行的 C语言 随机数
    /* 但是我不建议就这样运行，因为程序运行速度极快 */
    unsigned ret;
    srand(time(NULL));
    ret = rand() % 8;

    return ret;
}

// 训练主函数
int main(void)
{
    // 训练集，格式满足 Dark_Function(input[3k],input[3k+1],input[3k+2]) = output[2k]
    double input[24] = {0, 0, 0,
                        0, 0, 1,
                        0, 1, 0,
                        0, 1, 1,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        1, 1, 1};
    // 同时我们的训练集还要进行归一化
    /* 即 Xi = Xi/Σ(Xπi) , 其中 Xi ∈ X 是训练集中任意元素, Xπi 是所在行的全部元素 */

    // 新规则如下
    /* f(0, 0, 0) = 0  f(0, 0, 1) = 1
     * f(0, 1, 0) = 1  f(0, 1, 1) = 0
     * f(1, 0, 0) = 1  f(1, 0, 1) = 0
     * f(1, 1, 0) = 0  f(1, 1, 1) = 1
     */
    double output[16] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};

    xModel *pModel = (xModel *)malloc(sizeof(xModel));
    pvInit(&pModel);

    unsigned count = 0;

    /* 此代码为训练代码，暂时不需要 */
    // 因为 Keil MDK 不支持 Ctrl + / 所以这里采用了传统的斜杠和星号
    /*
    do
    {
        if (count % 20 == 0)
            printf("\n\nEpoch[%u]\n", ++count);

        unsigned i = 0;
        double loss = 0;
        for (; i < 4; ++i)
        {
            unsigned rdi = getRondomIndex();
            pvForward(&pModel, &input[3 * rdi], 3);
            loss += pvBackward(&pModel, LR * lr_fall(count), &output[2 * rdi], 2);
        }

        if (pModel->loss_end - pModel->loss == 4)
        {
            *(pModel->loss_end) = loss;
            pModel->loss_end = pModel->loss;
        }
        else
        {
            *(pModel->loss_end) = loss;
            ++(pModel->loss_end);
        }
        // 更新最佳 loss
        if (loss < pModel->best_loss)
        {
            pModel->best_loss = loss;
        }
        pvDisplayWeights(&pModel);

    } while (!pvEarlyStopDetect(&pModel) && count < EPOCH);
    */

    int ret[8];

    for (int i = 0; i < 8; ++i)
    {
        pvForward(&pModel, &input[3 * i], 3);
        ret[i] = ulGetResultIndex(&pModel);
    }

    printf("Return   ");
    for (int i = 0; i < 8; ++i)
        printf("%u ", ret[i]);
    printf("\nCritical 0 1 1 0 1 0 0 1\n");

    pvDeInit(&pModel);
    free(pModel);

    system("pause");
    return 0;
}
