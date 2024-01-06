# XOR_Experiment
An XOR expreiment used C and C++(Only use to fix network's weights)

## Branchs: main & deploy & expand

### main 分支
主要用于保存 PC 端运行代码，用于 XOR 网络的训练和调试

### deploy 分支
主要用于保存 STM32 侧的部署代码，用于 XOR 网络的部署

### expand 分支
主要用于我修改之后的代码，拿去当黑盒函数混答辩了（划掉）

### Tips
-Tip1：STM32F407VET6 训练还没测试，上次少 free 了三个导致内存泄漏，然后爆堆了-

-Tip2：STM32F103C8T6 我开着 wwdg 跑的，进入计算时会导致复位 MCU，不建议尝试-

（高速复位 MCU 非常的废板子，请珍爱与板子相遇的缘分）

![image](https://github.com/hhhhc-da/XOR_Experiment/blob/main/vs.png)
