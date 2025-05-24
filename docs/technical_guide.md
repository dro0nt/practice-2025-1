**Создание простой нейросети на Python: Техническое руководство для новичков**

**Введение**

Данное руководство описывает процесс создания и обучения простой нейросети на Python с использованием библиотек numpy и matplotlib. В нейросети реализованы популярные активационные функции (ReLU, sigmoid, tanh), процесс прямого и обратного распространения ошибки (backpropagation), а также визуализация процесса обучения.

Документ состоит из следующих разделов:

- Часть 1 — Предварительные требования
- Часть 2 — Пошаговые инструкции и код
- Часть 3 — Советы для новичков
- Часть 4 — Идеи для расширения проекта

Этот материал поможет новичкам понять базовые принципы работы нейросетей и научиться создавать простые модели с нуля.

-----
**Часть 1 — Предварительные требования**

- **Python 3.8+** — для запуска кода и работы с библиотеками.
- **Библиотеки:** numpy, matplotlib, scikit-learn.
- **Базовые знания Python:** массивы, функции, циклы.
-----
**Часть 2 — Пошаговые инструкции и код**

**Шаг 1: Импорт библиотек и настройка**



```
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
```

\# Установка генератора случайных чисел для воспроизводимости
```
np.random.seed(1)
```

**Шаг 2: Определение активационных функций и их производных**




```
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
```

**Шаг 3: Создание входных данных и выходов**



```

inputs = np.array([
[0, 0, 1],
[1, 1, 1],
[1, 0, 1],
[0, 1, 1]
])

outputs = np.array([[0], [1], [1], [0]])
```

**Шаг 4: Нормализация входных данных**

```

scaler = StandardScaler()

inputs = scaler.fit\_transform(inputs)
```

**Шаг 5: Инициализация весов нейросети**

```
def init_weights(in_dim, out_dim):
    return 2*np.random.random((in_dim, out_dim)) - 1

weights_0 = init_weights(3, 4)  # Входной слой -> скрытый слой 1

weights_1 = init_weights(4, 4)  # Скрытый слой 1 -> скрытый слой 2

weights_2 = init_weights(4, 1)  # Скрытый слой 2 -> выходной слой
```

**Шаг 6: Обучение нейросети с обратным распространением ошибки**



```
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

        # Прямое распространение (forward)

        l0 = x_batch

        l1 = relu(np.dot(l0, weights_0))

        l2 = tanh(np.dot(l1, weights_1))

        l3 = sigmoid(np.dot(l2, weights_2))

        # Ошибка

        error = y_batch - l3

        # Обратное распространение (backpropagation)

        l3_delta = error * sigmoid_derivative(l3)

        l2_error = l3_delta.dot(weights_2.T)

        l2_delta = l2_error * tanh_derivative(l2)

        l1_error = l2_delta.dot(weights_1.T)

        l1_delta = l1_error * relu_derivative(l1)

        # Обновление весов

        weights_2 += l2.T.dot(l3_delta)

        weights_1 += l1.T.dot(l2_delta)

        weights_0 += l0.T.dot(l1_delta)

        # Сохраняем ошибку для графика каждые 100 эпох

        if epoch % 100 == 0:

        mean_loss = np.mean(np.abs(error))

        losses.append(mean_loss)
```

**Шаг 7: Вывод результатов и визуализация обучения**



```

print("Обученные выходы:")

final_output = sigmoid(np.dot(tanh(np.dot(relu(np.dot(inputs, weights_0)), weights_1)), weights_2))

print(final_output)

plt.plot(range(0, epochs, 100), losses)

plt.title("Ошибка обучения")

plt.xlabel("Эпоха")

plt.ylabel("Средняя ошибка")

plt.grid(True)

plt.show()
```
-----
**Часть 3 — Советы для новичков**

- **Понимайте каждый шаг:** Разбирайтесь в функциях активации, прямом и обратном распространении.
- **Экспериментируйте:** Меняйте архитектуру (количество слоёв и нейронов), функцию активации.
- **Отслеживайте ошибки:** График обучения поможет понять, как улучшается модель.
- **Читай документацию:** Официальные гайды по numpy, matplotlib и sklearn помогут глубже понять инструменты.
-----
**Часть 4 — Возможности для расширения проекта**

- Добавить больше слоёв и нейронов.
- Реализовать разные функции потерь.
- Использовать библиотеки глубокого обучения (TensorFlow, PyTorch).
- Создать интерфейс для визуализации процесса обучения в реальном времени.
- Подключить датасеты из реального мира для решения практических задач.
-----
**Заключение**

Создание простой нейросети с нуля — отличный способ понять основы машинного обучения и работу нейронных сетей. Благодаря использованию Python и базовых библиотек вы можете экспериментировать, улучшать архитектуру и применять полученные знания для более сложных проектов в будущем.
