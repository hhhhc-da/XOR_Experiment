
#include "main.h"
#include "OLED.h"
// �ַ����ҾͲ�����
#include "OLED_Codetab.h"
// ͼƬ�⣨����GIF��
#include "BMP.h"

/*******************************************************************************
 * �� �� ��			: OLED_WriteByte
 * ��������			: OLED дһ���ֽ�
 * ע������			: �ȳ�ʼ�� IIC Ŷ
 *******************************************************************************/
void OLED_WriteByte(uint8_t addr, uint8_t data) {
    uint8_t TxData[2];

    // ���д������
    TxData[0] = addr; // ��ַ
    TxData[1] = data; // ����

    // ��������
    if (HAL_I2C_Master_Transmit(&hi2c1, OLED_ADDRESS, TxData, 2, HAL_MAX_DELAY) != HAL_OK) {
        // ����ʧ�ܴ���
    }
}

/*******************************************************************************
 * �� �� ��			: WriteCmd
 * ��������			: �� OLED д����
 * ע������			: ��
 *******************************************************************************/
void WriteCmd(unsigned char I2C_Cmd)
{
	OLED_WriteByte(0X00, I2C_Cmd); // ��0X00��ַ��ROM�д  51��Ƭ��LCD1602����
}

/*******************************************************************************
 * �� �� ��			: WriteData
 * ��������			: �� OLED д����
 * ע������			: ��
 *******************************************************************************/
void WriteData(unsigned char I2C_Data)
{
	OLED_WriteByte(0X40, I2C_Data); // 51��Ƭ��LCD1602����
}

/*******************************************************************************
 * �� �� ��			: OLED_Init
 * ��������			:  OELD ��Ļ��ʼ�����ⲿ�ִ����ǳ����ṩ��
 * ע������			: ��
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
	WriteCmd(0xff); // ���ȵ��� 0x00~0xff
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
 * �� �� ��			: OLED_SetPos
 * ��������			: �����������
 * ע������			: ��
 *******************************************************************************/
void OLED_SetPos(unsigned char x, unsigned char y)
{
	WriteCmd(0xb0 + y);				  // ��Щ���ǹ̶���  	�������ָ���
	WriteCmd((x & 0xf0) >> 4 | 0x10); // ȡ��4λ ָ����������
	WriteCmd((x & 0x0f) | 0x01);	  // ȡ��4λ
}

/*******************************************************************************
 * �� �� ��			: OLED_Full
 * ��������			: ȫ�����(һ��һ�е���䣬ÿ��8������)
 * ע������			: ��
 *******************************************************************************/
void OLED_Full(unsigned char Full_Data)
{
	unsigned char n, m;
	for (m = 0; m < 8; m++) // ��Ļ�ֳ���8ҳ ÿҳ��Ҫ����
	{
		WriteCmd(0xb0 + m); // 0xb0 �е�������� ������
		WriteCmd(0x00);
		WriteCmd(0x10);
		for (n = 0; n < 128; n++) // 128�� ÿ��8�����ص�д
		{
			WriteData(Full_Data);
		}
	}
}

/*******************************************************************************
 * �� �� ��			: OLED_Clear
 * ��������			: ��������������Ļʲô������ʾ
 * ע������			: ��
 *******************************************************************************/
void OLED_Clear(void)
{
	OLED_Full(0x00);
}

/*******************************************************************************
 * �� �� ��			: OLED_Open
 * ��������			: OLED ��(��ɱ�)
 * ע������			: ��
 *******************************************************************************/
void OLED_Open(void)
{
	WriteCmd(0x8D); // ���õ�ɱ�ָ�ָ������У�
	WriteCmd(0x14); // ������ɱ�
	WriteCmd(0xaf); // OLED����  ָ���Сд������ ��ú�ָ�����һ��
}

/*******************************************************************************
 * �� �� ��			: OLED_Close
 * ��������			: OLED �ر�(��ɱ�)
 * ע������			: ��
 *******************************************************************************/
void OLED_Close(void)
{
	WriteCmd(0x8D); // ���õ�ɱ�ָ�ָ������У�
	WriteCmd(0x10); // �رյ�ɱ�
	WriteCmd(0xAE); // oled�ر�
}

/*******************************************************************************
 * �� �� ��			: OLED_ShowStr
 * ��������			: OLED ��ʾ�ַ�����������ʽ��6*8  8*16��OLED_Codetab.h�ļ��
 * ע������			: �ȳ�ʼ�� IIC Ŷ
 *******************************************************************************/
void OLED_ShowStr(unsigned char x, unsigned char y, unsigned char ch[], unsigned TestSize)
{
	unsigned char c = 0, i = 0, j = 0;
	switch (TestSize)
	{
	case 1: // 6*8
	{
		while (ch[j] != '\0') // �ַ����������
		{
			c = ch[j] - 32; // ��Сд��ת��
			if (x > 126)	// XԽ���� �ŵ���һ��λ�� �ѵ���Ӧ����Y�� Y����ҳ X������
			{
				x = 0;
				y++; // �ص���һ�У���һ��
			}
			OLED_SetPos(x, y);
			for (i = 0; i < 6; i++)	   // 6��Ϊһ����ʾһ���ַ�
				WriteData(F6X8[c][i]); // ��ʾ����Ļ��
			x += 6;					   // ��ʽ��6*8  ��ת����һ���ַ� �ѵ���Ӧ����Y��
			j++;
		}
	}
	break;

	case 2: // 8*16
	{
		while (ch[j] != '\0')
		{
			c = ch[j] - 32;
			if (x > 120) // 8*16  120Խ�� û��
			{
				x = 0;
				y++; // �ص���һ�У���һ��
			}
			OLED_SetPos(x, y);
			for (i = 0; i < 8; i++) // ��16�ֳ��������֣� ��Ļ�Ƿֳ���8ҳ ������ʾ��Ŀǰ����̫����
				WriteData(F8X16[c * 16 + i]);
			OLED_SetPos(x, y + 1);
			for (i = 0; i < 8; i++)
				WriteData(F8X16[c * 16 + 8 + i]);
			x += 8; // ת����һ���ַ�
			j++;
		}
	}
	break;
	}
}

/*******************************************************************************
 * �� �� ��			: OLED_ShowCn
 * ��������			: OLED ��ʾ���ֺ�������ʽ��16*16���Լ�ȡģ��
 * ע������			: �ȳ�ʼ�� IIC Ŷ
 *******************************************************************************/
/*
void OLED_ShowCn(unsigned char x, unsigned char y, unsigned char N)
{
	unsigned char wn = 0;
	unsigned int addr = 32 * N; // һ��������32��16���Ʊ�ʾ��(.h�ļ�����)
	OLED_SetPos(x, y);

	for (wn = 0; wn < 16; wn++) // һ��������32��16���Ʊ�ʾ�� �ֳ���������
	{
		WriteData(F16X16[addr]);
		addr += 1;
	}
	OLED_SetPos(x, y + 1); // y+1 ����һҳд
	for (wn = 0; wn < 16; wn++)
	{
		WriteData(F16X16[addr]);
		addr += 1;
	}
}
*/
/*******************************************************************************
 * �� �� ��			: OLED_ShowBMP
 * ��������			: OLED ��ͼ
 * ע������			: ����ȫ��Ҫ��
 *******************************************************************************/
/* x ���÷�Χ 0 - 128��y ���÷�Χ 0 - 8����������Ҫ�ϸ��Ӧ */
/* �ڻ�ͼ������Image2Lcd������������ܸ����������� 128*64��֮��ת��Ϊ�ڰ�bmpͼ���ȡģ */
/* ʾ������: OLED_ShowBMP(96,0,128,2,power_4); */
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
	for (y = y0; y < y1; y++) // y�൱��ҳ�� һҳһҳ��  ÿһҳ��8��
	{
		OLED_SetPos(x0, y);
		for (x = x0; x < x1; x++)
		{
			WriteData(BMP[j++]);
		}
	}
}
/*******************************************************************************
 * �� �� ��			: OLED_DrawGIF
 * ��������			: OLED �� GIF
 * ע������			: ����ȫ��Ҫ��
 *******************************************************************************/
/*
 * @brief		��ʾ GIF
 * @param
 * 				x0����ʼ�е�ַ
 * 				y0����ʼҳ��ַ
 * 				x1����ֹ�е�ַ
 * 				y1����ֹҳ��ַ
 * 				k: ֡����
 * 				m: ��֡�����С
 * 				BMP����Ŷ�ͼ���������
 * p.s.
 * BMP Ϊһ�����ָ֡������飬ÿ��λ�ö���һ��֡��ָ��
 */

/* x ���÷�Χ 0 - 128��y ���÷�Χ 0 - 8����������Ҫ�ϸ��Ӧ */
/* �� IrfanView �����GIF���� */
/* ʾ������: OLED_DrawGIF(96,0,128,2,5,32*16,TEST_GIF); */
void OLED_DrawGIF(unsigned char x0, unsigned char y0,unsigned char x1, unsigned char y1, unsigned char k, int m, unsigned char* GIF[m])
{
	unsigned int j=0; //�������
 	unsigned char x,y,i; //�������
  
 	if(y1%8==0) y=y1/8;   //�ж���ֹҳ�Ƿ�Ϊ8��������
 	 else y=y1/8+1;
	for (i=0;i<k;i++) //�ӵ�һ֡��ʼ��
	{
		j = 0;
		for(y=y0;y<y1;y++) //����ʼҳ��ʼ��������ֹҳ
		{
			OLED_SetPos(x0,y); //��ҳ����ʼ�п�ʼ��
   			
			for(x=x0;x<x1;x++) //��x1 - x0 ��
	    	{
	    		WriteData(*(GIF[i]+j));	//��ͼƬ�ĵ�  
				j++;
	    	}
		}
		// �Լ��趨 GIF �ٶ�
		HAL_Delay(333);
	}
}
