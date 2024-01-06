/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "XOR_Model.h"
#include "AT24C256.h"
#include "OLED.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

// 学习率
#define LR 0.001
// 迭代最大轮数
#define EPOCH 5

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
xModel *pModel;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void saveModel(xModel **model);
void loadModel(xModel **model);

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */
  
  OLED_Init();
  
  OLED_Clear();
  
  
  // 训练集，格式满足 input[2k] XOR input[2k+1] = output[2k]
    double input[8] = {0, 0, 0, 1, 1, 1, 1, 0};
    // 1 0 表示 0 (index)，0 1 表示 1
    double output[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    pModel = (xModel *)malloc(sizeof(xModel));
    pvInit(&pModel);

	
	// 训练 17 次就会内存溢出
    unsigned count = 0;

    do
    {
        // if (count % 200 == 0)
        {
            // 每两百轮训练翻转一次电平
            printf("\r\n\r\nEpoch[%u]\r\n", ++count);
        }

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
        // 更新最佳 loss
        if (loss < pModel->best_loss)
        {
            pModel->best_loss = loss;
        }
        pvDisplayWeights(&pModel);

    } while (!pvEarlyStopDetect(&pModel) && count < EPOCH);
	
	
	printf("\r\n测试 %.3f XOR %.3f:\r\n",input[2],input[3]);
    pvForward(&pModel, &input[2], 2);
    int ret0 = ulGetResultIndex(&pModel);
    printf("获取结果为: %d\r\n\r\n", ret0);

    printf("测试 %.3f XOR %.3f:\r\n",input[6],input[7]);
    pvForward(&pModel, &input[6], 2);
    int ret1 = ulGetResultIndex(&pModel);
    printf("获取结果为: %d\r\n\r\n", ret1);

    printf("测试 %.3f XOR %.3f:\r\n",input[4],input[5]);
    pvForward(&pModel, &input[4], 2);
    int ret2 = ulGetResultIndex(&pModel);
    printf("获取结果为: %d\r\n\r\n", ret2);

    printf("测试 %.3f XOR %.3f:\r\n",input[0],input[1]);
    pvForward(&pModel, &input[0], 2);
    int ret3 = ulGetResultIndex(&pModel);
    printf("获取结果为: %d\r\n\r\n", ret3);
	
	double in[2] = {0, 0};
	
	// 保存模型
	saveModel(&pModel);
	
	HAL_Delay(100);
	OLED_ShowStr(20,2,(unsigned char*)"Computation", 1);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	HAL_Delay(100);
	in[0] = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_2);
	in[1] = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_3);
	  
	pvForward(&pModel, &in[0], 2);
	int ret = ulGetResultIndex(&pModel);
	if(ret == -1)
		printf("神经网络计算错误");
	else{
		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, ret);
		char str[20];
		sprintf(str, "%1d XOR %1d = %1d",(unsigned)in[0], (unsigned)in[1], ret);
		printf("%s\r\n", str);
		
		OLED_ShowStr(10,4,(unsigned char*)str, 1);
	}
  }

  pvDeInit(&pModel);
  free(pModel);
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 72;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void saveModel(xModel **model){
	// 写入数据格式 W11,W12,W21,W22,B1,B2,W11',W12',W21',W22',B1',B2'，写入到地址 0x00
	double_data_t *temp = (double_data_t*)malloc(sizeof(double_data_t));
	xModel *pTemp = *model;
	
	// 使用共用体指针批量存储
	temp->data = pTemp->W1[0];
	WriteAT24C256(0x00,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W1[1];
	WriteAT24C256(0x08,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W1[2];
	WriteAT24C256(0x10,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W1[3];
	WriteAT24C256(0x18,temp->byte,8);
	HAL_Delay(100);
	
	temp->data = pTemp->B1[0];
	WriteAT24C256(0x20,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->B1[1];
	WriteAT24C256(0x28,temp->byte,8);
	HAL_Delay(100);
	
	temp->data = pTemp->W2[0];
	WriteAT24C256(0x30,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W2[1];
	WriteAT24C256(0x38,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W2[2];
	WriteAT24C256(0x40,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->W2[3];
	WriteAT24C256(0x48,temp->byte,8);
	HAL_Delay(100);
	
	temp->data = pTemp->B2[0];
	WriteAT24C256(0x50,temp->byte,8);
	HAL_Delay(100);
	temp->data = pTemp->B2[1];
	WriteAT24C256(0x58,temp->byte,8);
	HAL_Delay(100);
	
	free(temp);
}

void loadModel(xModel **model){
	// 读取数据格式 W11,W12,W21,W22,B1,B2,W11',W12',W21',W22',B1',B2'，从地址 0x00 开始读取
	double_data_t *temp = (double_data_t*)malloc(sizeof(double_data_t));
	xModel *pTemp = *model;
	
	// 使用共用体指针批量存储
	ReadAT24C256(0x00,temp->byte,8);
	pTemp->W1[0] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x08,temp->byte,8);
	pTemp->W1[1] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x10,temp->byte,8);
	pTemp->W1[2] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x18,temp->byte,8);
	pTemp->W1[3] = temp->data;
	HAL_Delay(100);
	
	ReadAT24C256(0x20,temp->byte,8);
	pTemp->B1[0] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x28,temp->byte,8);
	pTemp->B1[1] = temp->data;
	HAL_Delay(100);
	
	ReadAT24C256(0x30,temp->byte,8);
	pTemp->W2[0] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x38,temp->byte,8);
	pTemp->W2[1] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x40,temp->byte,8);
	pTemp->W2[2] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x48,temp->byte,8);
	pTemp->W2[3] = temp->data;
	HAL_Delay(100);
	
	ReadAT24C256(0x50,temp->byte,8);
	pTemp->B2[0] = temp->data;
	HAL_Delay(100);
	ReadAT24C256(0x58,temp->byte,8);
	pTemp->B2[1] = temp->data;
	HAL_Delay(100);
	
	free(temp);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
