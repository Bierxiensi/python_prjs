import numpy as np
import matplotlib.pyplot as plt

def fun(theta, x):
    return theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[2]

def plot_regression_line(dataSet, theta):
    # 绘制原始数据点
    plt.scatter(dataSet[:, 0], dataSet[:, 2], color='blue', label='Data Points')

    # 构建预测线的端点
    x_values = [np.min(dataSet[:, 0] - 100), np.max(dataSet[:, 0] + 100)]
    y_values = theta[0] + theta[1] * x_values

    # 绘制拟合直线
    plt.plot(x_values, y_values, color='red', label='Regression Line')

    # 设置图表标题和标签
    plt.title('Linear Regression Fit')
    plt.xlabel('House Area (sq.ft)')
    plt.ylabel('House Price ($1000s)')
    plt.legend()

    # 显示图表
    plt.show()

def compute_cost(X, y, theta):
    """
    计算均方误差（MSE）损失函数
    """
    m = len(y)
    predictions = X.dot(theta)

    # print(10, m, predictions)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    """
    使用梯度下降法更新参数
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        #prediction = fun(theta, X[iterations])
        predictions = X.dot(theta)
        print(23,'X:', X, 'theta:', theta, 'predictions:', predictions)
        error = predictions - y
        print(25,'y:', y, 'error', error)
        gradient = X.T.dot(error) / m
        print(27,'gradient:', gradient, 'dot', X.T.dot(error))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def linear_regression(dataSet, alpha, iterations):
    """
    线性回归函数
    """
    # 从数据集中提取特征（房屋面积和房间数）和标签（房价）
    X = np.c_[np.ones(dataSet.shape[0]), dataSet[:, :2]]  # 特征矩阵
    y = dataSet[:, 2]  # 标签向量

    # 初始化参数 theta
    theta = [1, 0.2, 2]

    print(42, X, y, theta)
    # 执行梯度下降
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

    return theta, cost_history

# 给定的数据集
dataSet = np.array([
    [2104, 3, 400],
    [1600, 3, 330],
    [2400, 3, 369],
    [1416, 2, 342],
    [3000, 4, 540]
])

alpha = 0.0000003  # 学习率
iterations = 10  # 迭代次数

# 计算线性回归参数
result, cost_history = linear_regression(dataSet, alpha, iterations)
print("Theta参数：", result)


# 绘制成本函数随迭代次数变化的图像
plt.plot(range(iterations), cost_history)
plt.title('Cost Function Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

plot_regression_line(dataSet, result)
