import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd


user_file = "./AngularPlot/Data/video_0/user_1/pitch.csv" 
user_current = pd.read_csv(user_file)

# 绘制平滑的 CDF 曲线
sns.kdeplot(data=user_current['degree'], cumulative=True, label='User1')

# 添加标签和标题
plt.xlabel('pitch')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function')

# 显示图形
plt.show()
plt.show()