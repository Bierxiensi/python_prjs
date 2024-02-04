import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 计算损失函数
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

# 梯度下降算法
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = X.T.dot(error) / m
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# 计算损失函数的曲面
def plot_cost_surface(X, y):
    fig = plt.figure(figsize=(10, 8))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta = np.array([theta0, theta1])
            J_vals[i, j] = compute_cost(X, y, theta)

    theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = J_vals.T

    ax.plot_surface(theta0, theta1, J_vals, cmap='viridis')
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Cost')
    plt.title('Cost Function Surface')
    plt.show()

# 主函数
def main():
    dataSet = np.array([
        [2104, 3, 400],
        [1600, 3, 330],
        [2400, 3, 369],
        [1416, 2, 342],
        [3000, 4, 540]
    ])

    X = np.c_[np.ones(dataSet.shape[0]), dataSet[:, :2]]
    y = dataSet[:, 2]
    
    # 计算损失函数曲面
    plot_cost_surface(X, y)

    # 其他梯度下降和绘制拟合线的函数...

if __name__ == "__main__":
    main()
