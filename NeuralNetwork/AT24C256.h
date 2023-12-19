
#ifndef AT24C256_H
#define AT24C256_H

#include "i2c.h"
#include <stdlib.h>
#include "stdio.h"

// 方便移植
#define AT24C01 127
#define AT24C02 255
#define AT24C04 511
#define AT24C08 1023
#define AT24C16 2047
#define AT24C32 4095
#define AT24C64 8191
#define AT24C128 16383
#define AT24C256 32767

// 使用型号
#define EE_TYPE AT24C256

void AT24C256_WriteOneByte(uint16_t WriteAddr,uint8_t DataToWrite);
uint8_t AT24C256_ReadOneByte(uint16_t ReadAddr);
void WriteAT24C256(uint16_t addr, uint8_t *buf, uint16_t len);
void ReadAT24C256(uint16_t addr, uint8_t *buf, uint16_t len);
void pReadAT24C256MallocPointer(uint16_t addr, uint8_t **buf, uint16_t len);

typedef union{
  uint8_t byte[8];
  double data;
}double_data_t;

/*******************************************************************************
* 函 数 名			: pFreeAT24C256MallocPointer
* 函数功能			: 销毁申请的内存
* 注意事项			: 无
*******************************************************************************/
#define pFreeAT24C256MallocPointer( pxBuf ) \
					free( pxBuf )

#endif
