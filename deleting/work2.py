import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据集
data = {
    'Year': [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
    'Bus': [281516, 313296, 315576, 347969, 370640, 371822, 383161, 412590, 432021, 460970, 476255, 502916],
    'PGDP': [11232, 14368, 16738, 20505, 24121, 26222, 30876, 36403, 40007, 43852, 47203, 50251]
}

df = pd.DataFrame(data)

# 特征缩放
df_scaled = (df - df.mean()) / df.std()

# 添加截距项
df_scaled['Intercept'] = 1

# 划分训练集和测试集
train_data = df_scaled[df_scaled['Year'] <= 2004]
test_data = df_scaled[(df_scaled['Year'] >= 2005) & (df_scaled['Year'] <= 2007)]

# print(24, train_data, test_data)

# 选择特征和目标
X_train = train_data[['Intercept', 'Bus']].values
y_train = train_data['PGDP'].values

X_test = test_data[['Intercept', 'Bus']].values
y_test = test_data['PGDP'].values

# 定义梯度下降函数
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        # 计算预测值
        predictions = X.dot(theta)
        print(39, X, theta, predictions)
        # 计算误差
        errors = predictions - y
        
        # 更新参数
        theta = theta - (learning_rate / m) * X.T.dot(errors)
    
    return theta

# 初始化参数s
theta = [-306861.8208716349, 1.1225251621110288]

# 设置学习率和迭代次数
learning_rate = 0.1
iterations = 100

print(55, X_train, y_train, theta, learning_rate, iterations)
# 进行梯度下降
theta = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# 输出学到的参数
print("学到的参数：", theta)

# 在测试集上进行预测
predictions = X_test.dot(theta)
# 计算预测误差（均方根误差）
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("均方根误差（RMSE）：", rmse)

# 绘制拟合曲线和数据散点图
plt.scatter(df['Bus'], df['PGDP'], label='实际数据')
plt.plot(df['Bus'], df_scaled['Intercept'].values + df_scaled['Bus'].values * theta[1], color='red', label='拟合曲线')
plt.xlabel('Bus')
plt.ylabel('PGDP')
plt.legend()
plt.show()
