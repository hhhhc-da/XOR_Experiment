
#include "main.h"
#include "OLED.h"
// 字符库我就不改了
#include "OLED_Codetab.h"
// 图片库（包括GIF）
#include "BMP.h"

/*******************************************************************************
 * 函 数 名			: OLED_WriteByte
 * 函数功能			: OLED 写一个字节
 * 注意事项			: 先初始化 IIC 哦
 *******************************************************************************/
void OLED_WriteByte(uint8_t addr, uint8_t data) {
    uint8_t TxData[2];

    // 填充写入数据
    TxData[0] = addr; // 地址
    TxData[1] = data; // 数据

    // 发送数据
    if (HAL_I2C_Master_Transmit(&hi2c1, OLED_ADDRESS, TxData, 2, HAL_MAX_DELAY) != HAL_OK) {
        // 发送失败处理
    }
}

/*******************************************************************************
 * 函 数 名			: WriteCmd
 * 函数功能			: 给 OLED 写命令
 * 注意事项			: 无
 *******************************************************************************/
void WriteCmd(unsigned char I2C_Cmd)
{
	OLED_WriteByte(0X00, I2C_Cmd); // 在0X00地址（ROM里）写  51单片机LCD1602讲的
}

/*******************************************************************************
 * 函 数 名			: WriteData
 * 函数功能			: 给 OLED 写数据
 * 注意事项			: 无
 *******************************************************************************/
void WriteData(unsigned char I2C_Data)
{
	OLED_WriteByte(0X40, I2C_Data); // 51单片机LCD1602讲的
}

/*******************************************************************************
 * 函 数 名			: OLED_Init
 * 函数功能			:  OELD 屏幕初始化，这部分代码是厂家提供的
 * 注意事项			: 无
 *******************************************************************************/
void OLED_Init(void)
{
	HAL_Delay(100);
	WriteCmd(0xAE); // display off
	WriteCmd(0x20); // Set Memory Addressing Mode
	WriteCmd(0x10); // 00,Horizontal Addressing Mode;01,Vertical Addressing Mode;10,Page Addressing Mode (RESET);11,Invalid
	WriteCmd(0xb0); // Set Page Start Address for Page Addressing Mode,0-7
	WriteCmd(0xc8); // Set COM Output Scan Direction
	WriteCmd(0x00); //---set low column address
	WriteCmd(0x10); //---set high column address
	WriteCmd(0x40); //--set start line address
	WriteCmd(0x81); //--set contrast control register
	WriteCmd(0xff); // 亮度调节 0x00~0xff
	WriteCmd(0xa1); //--set segment re-map 0 to 127
	WriteCmd(0xa6); //--set normal display
	WriteCmd(0xa8); //--set multiplex ratio(1 to 64)
	WriteCmd(0x3F); //
	WriteCmd(0xa4); // 0xa4,Output follows RAM content;0xa5,Output ignores RAM content
	WriteCmd(0xd3); //-set display offset
	WriteCmd(0x00); //-not offset
	WriteCmd(0xd5); //--set display clock divide ratio/oscillator frequency
	WriteCmd(0xf0); //--set divide ratio
	WriteCmd(0xd9); //--set pre-charge period
	WriteCmd(0x22); //
	WriteCmd(0xda); //--set com pins hardware configuration
	WriteCmd(0x12);
	WriteCmd(0xdb); //--set vcomh
	WriteCmd(0x20); // 0x20,0.77xVcc
	WriteCmd(0x8d); //--set DC-DC enable
	WriteCmd(0x14); //
	WriteCmd(0xaf); //--turn on oled panel
}

/*******************************************************************************
 * 函 数 名			: OLED_SetPos
 * 函数功能			: 设置起点坐标
 * 注意事项			: 无
 *******************************************************************************/
void OLED_SetPos(unsigned char x, unsigned char y)
{
	WriteCmd(0xb0 + y);				  // 这些都是固定的  	看上面的指令表
	WriteCmd((x & 0xf0) >> 4 | 0x10); // 取高4位 指令表最后两行
	WriteCmd((x & 0x0f) | 0x01);	  // 取低4位
}

/*******************************************************************************
 * 函 数 名			: OLED_Full
 * 函数功能			: 全屏填充(一列一列地填充，每列8个像素)
 * 注意事项			: 无
 *******************************************************************************/
void OLED_Full(unsigned char Full_Data)
{
	unsigned char n, m;
	for (m = 0; m < 8; m++) // 屏幕分成了8页 每页都要处理
	{
		WriteCmd(0xb0 + m); // 0xb0 列的起点坐标 上面有
		WriteCmd(0x00);
		WriteCmd(0x10);
		for (n = 0; n < 128; n++) // 128列 每列8个像素地写
		{
			WriteData(Full_Data);
		}
	}
}

/*******************************************************************************
 * 函 数 名			: OLED_Clear
 * 函数功能			: 清屏函数，让屏幕什么都不显示
 * 注意事项			: 无
 *******************************************************************************/
void OLED_Clear(void)
{
	OLED_Full(0x00);
}

/*******************************************************************************
 * 函 数 名			: OLED_Open
 * 函数功能			: OLED 打开(电荷泵)
 * 注意事项			: 无
 *******************************************************************************/
void OLED_Open(void)
{
	WriteCmd(0x8D); // 设置电荷泵指令（指令表上有）
	WriteCmd(0x14); // 开启电荷泵
	WriteCmd(0xaf); // OLED唤醒  指令大小写都可以 最好和指令表上一致
}

/*******************************************************************************
 * 函 数 名			: OLED_Close
 * 函数功能			: OLED 关闭(电荷泵)
 * 注意事项			: 无
 *******************************************************************************/
void OLED_Close(void)
{
	WriteCmd(0x8D); // 设置电荷泵指令（指令表上有）
	WriteCmd(0x10); // 关闭电荷泵
	WriteCmd(0xAE); // oled关闭
}

/*******************************************************************************
 * 函 数 名			: OLED_ShowStr
 * 函数功能			: OLED 显示字符串函数，格式有6*8  8*16（OLED_Codetab.h文件里）
 * 注意事项			: 先初始化 IIC 哦
 *******************************************************************************/
void OLED_ShowStr(unsigned char x, unsigned char y, unsigned char ch[], unsigned TestSize)
{
	unsigned char c = 0, i = 0, j = 0;
	switch (TestSize)
	{
	case 1: // 6*8
	{
		while (ch[j] != '\0') // 字符串结束标记
		{
			c = ch[j] - 32; // 大小写的转化
			if (x > 126)	// X越界了 放到第一个位置 难道不应该是Y？ Y代表页 X代表列
			{
				x = 0;
				y++; // 回到第一行，下一列
			}
			OLED_SetPos(x, y);
			for (i = 0; i < 6; i++)	   // 6列为一组显示一个字符
				WriteData(F6X8[c][i]); // 显示到屏幕上
			x += 6;					   // 格式：6*8  跳转到下一个字符 难道不应该是Y？
			j++;
		}
	}
	break;

	case 2: // 8*16
	{
		while (ch[j] != '\0')
		{
			c = ch[j] - 32;
			if (x > 120) // 8*16  120越界 没懂
			{
				x = 0;
				y++; // 回到第一行，下一列
			}
			OLED_SetPos(x, y);
			for (i = 0; i < 8; i++) // 把16分成两个部分， 屏幕是分成了8页 上下显示（目前还不太懂）
				WriteData(F8X16[c * 16 + i]);
			OLED_SetPos(x, y + 1);
			for (i = 0; i < 8; i++)
				WriteData(F8X16[c * 16 + 8 + i]);
			x += 8; // 转到下一个字符
			j++;
		}
	}
	break;
	}
}

/*******************************************************************************
 * 函 数 名			: OLED_ShowCn
 * 函数功能			: OLED 显示汉字函数，格式有16*16（自己取模）
 * 注意事项			: 先初始化 IIC 哦
 *******************************************************************************/
/*
void OLED_ShowCn(unsigned char x, unsigned char y, unsigned char N)
{
	unsigned char wn = 0;
	unsigned int addr = 32 * N; // 一个字是由32个16进制表示的(.h文件两行)
	OLED_SetPos(x, y);

	for (wn = 0; wn < 16; wn++) // 一个字是由32个16进制表示的 分成两个部分
	{
		WriteData(F16X16[addr]);
		addr += 1;
	}
	OLED_SetPos(x, y + 1); // y+1 在下一页写
	for (wn = 0; wn < 16; wn++)
	{
		WriteData(F16X16[addr]);
		addr += 1;
	}
}
*/
/*******************************************************************************
 * 函 数 名			: OLED_ShowBMP
 * 函数功能			: OLED 画图
 * 注意事项			: 下面全是要求
 *******************************************************************************/
/* x 可用范围 0 - 128，y 可用范围 0 - 8，像素总数要严格对应 */
/* 在画图（或者Image2Lcd）里调整像素总个数，不高于 128*64，之后转换为黑白bmp图像后取模 */
/* 示例代码: OLED_ShowBMP(96,0,128,2,power_4); */
void OLED_ShowBMP(unsigned char x0, unsigned char y0, unsigned char x1, unsigned char y1, unsigned char BMP[])
{
	unsigned int j = 0;
	unsigned char x, y;
	if (y1 % 8 == 0)
	{
		y = y1 / 8;
	}
	else
	{
		y = y1 / 8 + 1;
	}
	for (y = y0; y < y1; y++) // y相当于页数 一页一页的  每一页有8行
	{
		OLED_SetPos(x0, y);
		for (x = x0; x < x1; x++)
		{
			WriteData(BMP[j++]);
		}
	}
}
/*******************************************************************************
 * 函 数 名			: OLED_DrawGIF
 * 函数功能			: OLED 画 GIF
 * 注意事项			: 下面全是要求
 *******************************************************************************/
/*
 * @brief		显示 GIF
 * @param
 * 				x0：起始列地址
 * 				y0：起始页地址
 * 				x1：终止列地址
 * 				y1：终止页地址
 * 				k: 帧个数
 * 				m: 单帧数组大小
 * 				BMP：存放动图代码的数组
 * p.s.
 * BMP 为一个存放帧指针的数组，每个位置都放一个帧的指针
 */

/* x 可用范围 0 - 128，y 可用范围 0 - 8，像素总数要严格对应 */
/* 在 IrfanView 里调整GIF属性 */
/* 示例代码: OLED_DrawGIF(96,0,128,2,5,32*16,TEST_GIF); */
void OLED_DrawGIF(unsigned char x0, unsigned char y0,unsigned char x1, unsigned char y1, unsigned char k, int m, unsigned char* GIF[m])
{
	unsigned int j=0; //定义变量
 	unsigned char x,y,i; //定义变量
  
 	if(y1%8==0) y=y1/8;   //判断终止页是否为8的整数倍
 	 else y=y1/8+1;
	for (i=0;i<k;i++) //从第一帧开始画
	{
		j = 0;
		for(y=y0;y<y1;y++) //从起始页开始，画到终止页
		{
			OLED_SetPos(x0,y); //在页的起始列开始画
   			
			for(x=x0;x<x1;x++) //画x1 - x0 列
	    	{
	    		WriteData(*(GIF[i]+j));	//画图片的点  
				j++;
	    	}
		}
		// 自己设定 GIF 速度
		HAL_Delay(333);
	}
}
