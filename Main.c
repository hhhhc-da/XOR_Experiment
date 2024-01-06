#include "Model.h"

// �����ܸ���
#define GAUSS_NUM 26
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

unsigned getRondomIndex(void)
{
    /* ���� systick ��ȡӲ�������ʲô�� */
    // �����ʹ�ÿ����� PC ���е� C���� �����
    /* �����Ҳ�������������У���Ϊ���������ٶȼ��� */
    unsigned ret;
    srand(time(NULL));
    ret = rand() % 8;

    return ret;
}

// ѵ��������
int main(void)
{
    // ѵ��������ʽ���� Dark_Function(input[3k],input[3k+1],input[3k+2]) = output[2k]
    double input[24] = {0, 0, 0,
                        0, 0, 1,
                        0, 1, 0,
                        0, 1, 1,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        1, 1, 1};
    // ͬʱ���ǵ�ѵ������Ҫ���й�һ��
    /* �� Xi = Xi/��(X��i) , ���� Xi �� X ��ѵ����������Ԫ��, X��i �������е�ȫ��Ԫ�� */

    // �¹�������
    /* f(0, 0, 0) = 0  f(0, 0, 1) = 1
     * f(0, 1, 0) = 1  f(0, 1, 1) = 0
     * f(1, 0, 0) = 1  f(1, 0, 1) = 0
     * f(1, 1, 0) = 0  f(1, 1, 1) = 1
     */
    double output[16] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};

    xModel *pModel = (xModel *)malloc(sizeof(xModel));
    pvInit(&pModel);

    unsigned count = 0;

    /* �˴���Ϊѵ�����룬��ʱ����Ҫ */
    // ��Ϊ Keil MDK ��֧�� Ctrl + / ������������˴�ͳ��б�ܺ��Ǻ�
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
        // ������� loss
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
