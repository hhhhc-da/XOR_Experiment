#ifndef OLED_H
#define OLED_H

#include "i2c.h"
#include "stdint.h"

extern const unsigned char F6X8[][6];
extern const unsigned char F8X16[];

#define  OLED_ADDRESS 0x78 //这个地址厂家设置好的？


void OLED_WriteByte(uint8_t addr , uint8_t data);
void WriteCmd(unsigned char I2C_Cmd);
void WriteData(unsigned char I2C_Data);
void OLED_Init(void);
void OLED_SetPos(unsigned char x, unsigned char y);
void OLED_Full(unsigned char Full_Data);
void OLED_Clear(void);
void OLED_Open(void);
void OLED_Close(void);
void OLED_ShowStr(unsigned char x, unsigned char y , unsigned char ch[] , unsigned TestSize);
void OLED_ShowCn(unsigned char x, unsigned char y , unsigned char N);
void OLED_ShowBMP(unsigned char x0, unsigned char y0, unsigned char x1, unsigned char y1, unsigned char BMP[]);
void OLED_DrawGIF(unsigned char x0, unsigned char y0,unsigned char x1, unsigned char y1, unsigned char k, int m, unsigned char* GIF[m]);

extern uint8_t* TEST_GIF[];
extern uint8_t power_4[];
extern uint8_t power_3[];
extern uint8_t power_2[];
extern uint8_t power_1[];
extern uint8_t power_0[];

#endif
