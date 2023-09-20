使用梯度下降算法来找到函数f(x) = (x - 1)^2的最小值点
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x - 1) ** 2


def df(x):
    return 2 * (x - 1)


def gradient_descent(learning_rate, num_iterations):
    x = 0  # 初始点
    history = []  # 用于存储每次迭代的点和损失值

    for _ in range(num_iterations):
        gradient = df(x)
        x -= learning_rate * gradient
        loss = f(x)
        history.append((x, loss))

    return history


learning_rate = 0.1
num_iterations = 20


history = gradient_descent(learning_rate, num_iterations)


x_values, loss_values = zip(*history)


x = np.linspace(-1, 3, 400)
plt.plot(x, f(x), label='f(x) = (x - 1)^2')
plt.scatter(x_values, loss_values, c='red', marker='o', label='Gradient Descent')


plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent for f(x)')
plt.legend()


plt.grid()
plt.show()





使用梯度下降法来寻找函数$f(x, y) = (x - 2)^2 + (y - 3)^2$的最小值点
import numpy as np
import matplotlib.pyplot as plt


def target_function(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2


def gradient(x, y):
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y - 3)
    return grad_x, grad_y


x, y = 0.0, 0.0
learning_rate = 0.1
num_iterations = 100


x_history, y_history = [], []

for i in range(num_iterations):
    grad_x, grad_y = gradient(x, y)
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y
    x_history.append(x)
    y_history.append(y)


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


使用梯度下降法来寻找函数$f(x, y) = (x - 2)^2 + (y - 3)^2$的最小值点
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]



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



theta = np.random.randn(2, 1)
learning_rate = 0.1
num_iterations = 1000


theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, num_iterations)

plt.scatter(X, y, alpha=0.6, label='Data Points')
plt.plot(X, X_b.dot(theta), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Gradient Descent')


plt.figure()
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.show()


print("线性回归参数:")
print(f"截距 (theta0): {theta[0][0]:.2f}")
print(f"斜率 (theta1): {theta[1][0]:.2f}")
