# FoV_Simulator
全景视频视野落点仿真模拟

## Code
- convlstm.py: 模型代码
- data_process: 数据处理，四元数计算等工具方法
- evaluate.py：验证和绘图
- fov_predict: 由真实数据利用预测模型输出预测数据
- test.py: 方法测试

## Data
- heatmap：视野区域图
- model：训练好的模型
- Predict：预测视野数据
- Real：处理后的视野数据
- SalData：显著性数据数据集
- vr-dataset：视野的原数据

## Experiment:
- 根目录内的是整体的视野落点仿真，支持后续算法接入
- AngularPlot：不同用户、不同视频中角速度、俯仰角、偏航角的CDF图仿真实验