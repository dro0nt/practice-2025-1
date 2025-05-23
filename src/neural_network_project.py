import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# 1. Загрузка и предобработка данных
max_words = 10000  # Размер словаря
max_len = 200  # Максимальная длина отзыва

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_words)

# 2. Padding для одинаковой длины последовательностей
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)


# 3. Слой Attention
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)


# 4. Создание модели
embedding_dim = 128

model = models.Sequential([
    layers.Embedding(max_words, embedding_dim),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    AttentionLayer(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Обучение модели
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Оценка модели
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")


# 7. Функция для предсказания тональности
def predict_sentiment(text):
    # Загрузка словаря IMDB
    word_index = datasets.imdb.get_word_index()

    # Очистка текста
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = text.split()

    # Векторизация текста
    text_seq = [word_index[word] + 3 for word in words
                if word in word_index and word_index[word] < max_words - 3]
    text_pad = pad_sequences([text_seq], maxlen=max_len)

    # Предсказание
    prediction = model.predict(text_pad)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"


# Тестирование
print("\nПримеры предсказаний:")
print(predict_sentiment("This movie was great and I loved it!"))  # Positive
print(predict_sentiment("Terrible film, waste of time."))  # Negative
print(predict_sentiment("The plot was boring but actors were good."))  # ?