# XOR_Experiment
An XOR expreiment used C and C++(Only use to fix network's weights)

## Branchs: main & deploy

main 分支主要用于保存 PC 端运行代码，用于 XOR 网络的训练和调试

deploy 分支主要用于保存 STM32 侧的部署代码，用于 XOR 网络的部署

（Tips：STM32F407VET6 只能跑17次训练，之后就会数据全0，应该是 malloc 没有申请到空间）

![image](https://github.com/hhhhc-da/XOR_Experiment/blob/main/vs.png)
