import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def target_function(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2

# 定义梯度函数
def gradient(x, y):
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y - 3)
    return grad_x, grad_y

# 初始参数和学习率
x, y = 0.0, 0.0
learning_rate = 0.1
num_iterations = 100

# 用于保存每次迭代的结果
x_history, y_history = [], []

# 梯度下降优化过程
for i in range(num_iterations):
    grad_x, grad_y = gradient(x, y)
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y
    x_history.append(x)
    y_history.append(y)

# 绘制函数曲面
x_range = np.linspace(-1, 5, 100)
y_range = np.linspace(-1, 7, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = target_function(X, Y)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.linspace(0, 20, 50), cmap='viridis')
plt.plot(x_history, y_history, marker='o', color='red', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent Optimization')
plt.colorbar()
plt.show()

print(f"最小值点：({x:.2f}, {y:.2f})")
print(f"最小值：{target_function(x, y):.2f}")
