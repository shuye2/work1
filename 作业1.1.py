import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return (x - 1) ** 2

# 定义目标函数的导数
def df(x):
    return 2 * (x - 1)

# 梯度下降函数
def gradient_descent(learning_rate, num_iterations):
    x = 0  # 初始点
    history = []  # 用于存储每次迭代的点和损失值

    for _ in range(num_iterations):
        gradient = df(x)
        x -= learning_rate * gradient
        loss = f(x)
        history.append((x, loss))

    return history

# 设置学习率和迭代次数
learning_rate = 0.1
num_iterations = 20

# 运行梯度下降算法
history = gradient_descent(learning_rate, num_iterations)

# 提取迭代过程中的点和损失值
x_values, loss_values = zip(*history)

# 绘制函数图像
x = np.linspace(-1, 3, 400)
plt.plot(x, f(x), label='f(x) = (x - 1)^2')
plt.scatter(x_values, loss_values, c='red', marker='o', label='Gradient Descent')

# 添加标签和图例
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent for f(x)')
plt.legend()

# 显示图形
plt.grid()
plt.show()
