import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置项 x0 = 1 到特征矩阵 X
X_b = np.c_[np.ones((100, 1)), X]


# 定义梯度下降函数
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        error = X.dot(theta) - y
        gradient = (1 / m) * X.T.dot(error)
        theta -= learning_rate * gradient
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

    return theta, cost_history


# 初始化参数和学习率
theta = np.random.randn(2, 1)
learning_rate = 0.1
num_iterations = 1000

# 调用梯度下降函数拟合线性回归模型
theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, num_iterations)

# 绘制数据散点图和线性回归线
plt.scatter(X, y, alpha=0.6, label='Data Points')
plt.plot(X, X_b.dot(theta), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Gradient Descent')

# 绘制代价函数的收敛曲线
plt.figure()
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.show()

# 打印线性回归参数
print("线性回归参数:")
print(f"截距 (theta0): {theta[0][0]:.2f}")
print(f"斜率 (theta1): {theta[1][0]:.2f}")
