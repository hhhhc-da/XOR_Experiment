#include "XOR_Model.h"

// ѧϰ��
#define LR 0.001
// �����������
#define EPOCH 1000

// ѵ��������
int main(void)
{
    // ѵ��������ʽ���� input[2k] XOR input[2k+1] = output[2k]
    double input[8] = {0, 0, 0, 1, 1, 1, 1, 0};
    // 1 0 ��ʾ 0 (index)��0 1 ��ʾ 1
    double output[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    xModel *pModel = (xModel *)malloc(sizeof(xModel));
    pvInit(&pModel);

    unsigned count = 0;

    do
    {
        if (count % 200 == 0)
            printf("\n\nEpoch[%u]\n", ++count);

        unsigned i = 0;
        double loss = 0;
        for (; i < 4; ++i)
        {
            pvForward(&pModel, &input[2 * i], 2);
            loss += pvBackward(&pModel, LR * lr_fall(count), &output[2 * i], 2);
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
        // ������� loss
        if (loss < pModel->best_loss)
        {
            pModel->best_loss = loss;
        }
        pvDisplayWeights(&pModel);

    } while (!pvEarlyStopDetect(&pModel) && count < EPOCH);

    printf("\n���� 0 XOR 1:\n");
    pvForward(&pModel, &input[2], 2);
    int ret0 = ulGetResultIndex(&pModel);
    printf("��ȡ���Ϊ: %d\n\n", ret0);

    printf("���� 1 XOR 1:\n");
    pvForward(&pModel, &input[4], 2);
    int ret1 = ulGetResultIndex(&pModel);
    printf("��ȡ���Ϊ: %d\n\n", ret1);

    printf("���� 1 XOR 0:\n");
    pvForward(&pModel, &input[6], 2);
    int ret2 = ulGetResultIndex(&pModel);
    printf("��ȡ���Ϊ: %d\n\n", ret2);

    printf("���� 0 XOR 0:\n");
    pvForward(&pModel, &input[0], 2);
    int ret3 = ulGetResultIndex(&pModel);
    printf("��ȡ���Ϊ: %d\n\n\n", ret3);

    free(pModel);

    system("pause");
    return 0;
}
