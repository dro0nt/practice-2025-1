import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Установка генератора случайных чисел для воспроизводимости
np.random.seed(1)


# ==== Активационные функции ====
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ==== Данные ====
inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

outputs = np.array([[0], [1], [1], [0]])

# ==== Нормализация ====
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)


# ==== Инициализация весов ====
def init_weights(in_dim, out_dim):
    return 2 * np.random.random((in_dim, out_dim)) - 1


weights_0 = init_weights(3, 4)  # Вход -> скрытый слой 1
weights_1 = init_weights(4, 4)  # скрытый слой 1 -> скрытый слой 2
weights_2 = init_weights(4, 1)  # скрытый слой 2 -> выход

# ==== Обучение ====
epochs = 10000
batch_size = 2
losses = []

for epoch in range(epochs):
    perm = np.random.permutation(len(inputs))
    inputs_shuffled = inputs[perm]
    outputs_shuffled = outputs[perm]

    for i in range(0, len(inputs), batch_size):
        x_batch = inputs_shuffled[i:i + batch_size]
        y_batch = outputs_shuffled[i:i + batch_size]

        # Forward
        l0 = x_batch
        l1 = relu(np.dot(l0, weights_0))
        l2 = tanh(np.dot(l1, weights_1))
        l3 = sigmoid(np.dot(l2, weights_2))

        # Ошибка
        error = y_batch - l3

        # Backpropagation
        l3_delta = error * sigmoid_derivative(l3)
        l2_error = l3_delta.dot(weights_2.T)
        l2_delta = l2_error * tanh_derivative(l2)
        l1_error = l2_delta.dot(weights_1.T)
        l1_delta = l1_error * relu_derivative(l1)

        # Обновление весов
        weights_2 += l2.T.dot(l3_delta)
        weights_1 += l1.T.dot(l2_delta)
        weights_0 += l0.T.dot(l1_delta)

    # Сохраняем ошибку на каждой эпохе
    if epoch % 100 == 0:
        mean_loss = np.mean(np.abs(error))
        losses.append(mean_loss)

# ==== Вывод результатов ====
print("Обученные выходы:")
final_output = sigmoid(np.dot(tanh(np.dot(relu(np.dot(inputs, weights_0)), weights_1)), weights_2))
print(final_output)

# ==== Визуализация ====
plt.plot(range(0, epochs, 100), losses)
plt.title("Ошибка обучения")
plt.xlabel("Эпоха")
plt.ylabel("Средняя ошибка")
plt.grid(True)
plt.show()
