
#include "AT24C256.h"

/*******************************************************************************
* �� �� ��			: AT24C256_WriteOneByte
* ��������			: ���� IIC д��ĳ��ַ�ϵ�ĳһ���ֽ�
* ע������			: ��
*******************************************************************************/
void AT24C256_WriteOneByte(uint16_t WriteAddr, uint8_t DataToWrite) {
    uint16_t DevAddress = 0;
    uint8_t TxData[3];

    // ����EE_TYPEѡ���豸��ַ
    if (EE_TYPE > AT24C16) {
        DevAddress = 0xA0;
    } else {
        DevAddress = 0xA0 + ((WriteAddr / 256) << 1);
    }

    // ���д������
    TxData[0] = WriteAddr >> 8; // �ߵ�ַ
    TxData[1] = WriteAddr % 256; // �͵�ַ
    TxData[2] = DataToWrite; // ����

    // ��������
    if (HAL_I2C_Master_Transmit(&hi2c1, DevAddress, TxData, 3, HAL_MAX_DELAY) != HAL_OK) {
        // ����ʧ�ܴ���
		printf("I2C�������ݷ���ʧ��");
    }

    // ��ʱ
    HAL_Delay(10);
}

/*******************************************************************************
* �� �� ��			: AT24C256_ReadOneByte
* ��������			: ���� IIC ��ȡĳ��ַ�ϵ�ĳһ���ֽ�
* ע������			: ��
*******************************************************************************/
uint8_t AT24C256_ReadOneByte(uint16_t ReadAddr) {
    uint8_t temp = 0;
    uint16_t DevAddress = 0;

    // ����EE_TYPEѡ���豸��ַ
    if (EE_TYPE > AT24C16) {
        DevAddress = 0xA0;
    } else {
        DevAddress = 0xA0 + ((ReadAddr / 256) << 1);
    }

    // ���Ͷ�ȡ��ַ
    if (HAL_I2C_Master_Transmit(&hi2c1, DevAddress, (uint8_t*)&ReadAddr, 2, HAL_MAX_DELAY) != HAL_OK) {
        // ����ʧ�ܴ���
		printf("I2C��ȡ��ַ����ʧ��");
    }

    // ��������
    if (HAL_I2C_Master_Receive(&hi2c1, DevAddress, &temp, 1, HAL_MAX_DELAY) != HAL_OK) {
        // ����ʧ�ܴ���
		printf("I2C��ȡ���ݽ���ʧ��");
    }

    return temp;
}

/*******************************************************************************
* �� �� ��			: WriteAT24C256
* ��������			: ���� IIC ����д��ĳ��ַ
* ע������			: ��
*******************************************************************************/
void WriteAT24C256(uint16_t addr, uint8_t *buf, uint16_t len) {
	while(len--) {
		AT24C256_WriteOneByte(addr, *buf);
		addr++;
		buf++;
		
	}
}

/*******************************************************************************
* �� �� ��			: ReadAT24C256
* ��������			: ���� IIC ������ȡĳ��ַ
* ע������			: ����ֱ����ָ���ַ����
*******************************************************************************/
void ReadAT24C256(uint16_t addr, uint8_t *buf, uint16_t len) {
	while(len--) {
		*buf++ = AT24C256_ReadOneByte(addr++);
	}
}

/*******************************************************************************
* �� �� ��			: pReadAT24C256MallocPointer
* ��������			: ���� IIC ������ȡĳ��ַ
* ע������			: �ǵ�����֮�󽻻��ڴ�
*******************************************************************************/
void pReadAT24C256MallocPointer(uint16_t addr, uint8_t **buf, uint16_t len) {
	uint8_t *pTmp = (uint8_t *)malloc(len), *pTmp2 = pTmp; 
	while(len--) 
		*pTmp2++ = AT24C256_ReadOneByte(addr++);
	*buf = pTmp;
}
