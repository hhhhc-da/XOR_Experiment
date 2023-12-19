
#include "AT24C256.h"

/*******************************************************************************
* 函 数 名			: AT24C256_WriteOneByte
* 函数功能			: 利用 IIC 写入某地址上的某一个字节
* 注意事项			: 无
*******************************************************************************/
void AT24C256_WriteOneByte(uint16_t WriteAddr, uint8_t DataToWrite) {
    uint16_t DevAddress = 0;
    uint8_t TxData[3];

    // 根据EE_TYPE选择设备地址
    if (EE_TYPE > AT24C16) {
        DevAddress = 0xA0;
    } else {
        DevAddress = 0xA0 + ((WriteAddr / 256) << 1);
    }

    // 填充写入数据
    TxData[0] = WriteAddr >> 8; // 高地址
    TxData[1] = WriteAddr % 256; // 低地址
    TxData[2] = DataToWrite; // 数据

    // 发送数据
    if (HAL_I2C_Master_Transmit(&hi2c1, DevAddress, TxData, 3, HAL_MAX_DELAY) != HAL_OK) {
        // 发送失败处理
		printf("I2C发送数据发送失败");
    }

    // 延时
    HAL_Delay(10);
}

/*******************************************************************************
* 函 数 名			: AT24C256_ReadOneByte
* 函数功能			: 利用 IIC 读取某地址上的某一个字节
* 注意事项			: 无
*******************************************************************************/
uint8_t AT24C256_ReadOneByte(uint16_t ReadAddr) {
    uint8_t temp = 0;
    uint16_t DevAddress = 0;

    // 根据EE_TYPE选择设备地址
    if (EE_TYPE > AT24C16) {
        DevAddress = 0xA0;
    } else {
        DevAddress = 0xA0 + ((ReadAddr / 256) << 1);
    }

    // 发送读取地址
    if (HAL_I2C_Master_Transmit(&hi2c1, DevAddress, (uint8_t*)&ReadAddr, 2, HAL_MAX_DELAY) != HAL_OK) {
        // 发送失败处理
		printf("I2C读取地址发送失败");
    }

    // 接收数据
    if (HAL_I2C_Master_Receive(&hi2c1, DevAddress, &temp, 1, HAL_MAX_DELAY) != HAL_OK) {
        // 接收失败处理
		printf("I2C读取数据接收失败");
    }

    return temp;
}

/*******************************************************************************
* 函 数 名			: WriteAT24C256
* 函数功能			: 利用 IIC 连续写入某地址
* 注意事项			: 无
*******************************************************************************/
void WriteAT24C256(uint16_t addr, uint8_t *buf, uint16_t len) {
	while(len--) {
		AT24C256_WriteOneByte(addr, *buf);
		addr++;
		buf++;
		
	}
}

/*******************************************************************************
* 函 数 名			: ReadAT24C256
* 函数功能			: 利用 IIC 连续读取某地址
* 注意事项			: 请勿直接用指针地址接收
*******************************************************************************/
void ReadAT24C256(uint16_t addr, uint8_t *buf, uint16_t len) {
	while(len--) {
		*buf++ = AT24C256_ReadOneByte(addr++);
	}
}

/*******************************************************************************
* 函 数 名			: pReadAT24C256MallocPointer
* 函数功能			: 利用 IIC 连续读取某地址
* 注意事项			: 记得用完之后交还内存
*******************************************************************************/
void pReadAT24C256MallocPointer(uint16_t addr, uint8_t **buf, uint16_t len) {
	uint8_t *pTmp = (uint8_t *)malloc(len), *pTmp2 = pTmp; 
	while(len--) 
		*pTmp2++ = AT24C256_ReadOneByte(addr++);
	*buf = pTmp;
}
