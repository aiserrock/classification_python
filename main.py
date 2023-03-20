import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    tf.compat.v1.disable_eager_execution()

    # Загружаем датасет
    text_rumor_df = pd.read_csv('datasets/data.csv', encoding='latin-1')
    le = LabelEncoder()
    x = text_rumor_df['text'].to_numpy()
    y = le.fit_transform(list(text_rumor_df['is_rumor']))

    def get_index(percent):
        dataset_size = text_rumor_df.shape[0]
        return int(dataset_size * percent)

    # Выделяем выборки для обучения
    train_index = get_index(0.4)
    x_train = np.asarray(x[:train_index])
    y_train = np.asarray(y[:train_index])

    # Выделяем выборки для валидации
    val_index = get_index(0.3) + train_index
    x_val = np.asarray(x[train_index:val_index])
    y_val = np.asarray(y[train_index:val_index])

    # Выделяем выборки для теста
    x_test = np.asarray(x[val_index:])
    y_test = np.asarray(y[val_index:])

    # Языковая модель ELMO от tensorflow_hub
    elmo = "https://tfhub.dev/google/elmo/3"
    # берём слой языковой модели на архитектуре ELMO
    hub_layer = hub.KerasLayer(elmo, input_shape=[], dtype=tf.string, trainable=False)
    hub_layer(x_train)

    # Модель
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # Функция потерь и оптимизатор
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Обучаем модель
    history = model.fit(x_train,
                        y_train,
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        epochs=2,
                        verbose=1, )
    model.save('ELMoModel.keras')
    # model = tf.keras.saving.load_model('ELMoModel.keras')

    # Оценка модели
    results = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    # Графки зависимостей epoch/accuracy
    history_dict = history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'b', label='Training acc')
    plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # График зависимости epoch/loss
    loss_values = history_dict['loss']
    val_los_values = history_dict['val_loss']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_los_values, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Пробуем распознавание
    predicts = model.predict(x_test)
    y_pred = np.concatenate(np.around(predicts), axis=0)

    # Оценка результата классификации с помощью метрик
    print(metrics.classification_report(y_test, y_pred, target_names=["rumor", "not a rumor"]))


if __name__ == '__main__':
    main()
