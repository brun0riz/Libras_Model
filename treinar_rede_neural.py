import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregamento de dados com augmentação opcional ---
def load_sequential_data(data_path, augment=False, augment_factor=1):
    sequences, labels = [], []
    actions = sorted([a for a in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, a)) and not a.startswith('.')])
    label_map = {label: i for i, label in enumerate(actions)}

    def augment_sequence(seq):
        noise = np.random.normal(0, 0.01, seq.shape)
        return np.clip(seq + noise, -1.0, 1.0)

    for action in actions:
        path = os.path.join(data_path, action)
        for file in os.listdir(path):
            if file.endswith('.npy'):
                try:
                    arr = np.load(os.path.join(path, file))
                    if arr.shape == (30, 42):
                        sequences.append(arr)
                        labels.append(label_map[action])
                        if augment:
                            for _ in range(augment_factor):
                                sequences.append(augment_sequence(arr))
                                labels.append(label_map[action])
                except Exception as e:
                    print(f"Erro ao carregar {file}: {e}")

    return np.array(sequences), np.array(labels), actions

# --- 2. Definição do modelo para Keras Tuner ---
def build_model(hp):
    l2_val = hp.Choice('l2_reg', [0.01, 0.001, 0.0001])
    model = keras.Sequential([
        keras.Input(shape=(30, 42)),
        keras.layers.LSTM(
            units=hp.Int('units_lstm_1', 32, 128, step=32),
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(l2_val)
        ),
        keras.layers.Dropout(hp.Float('dropout_1', 0.3, 0.6, step=0.1)),
        keras.layers.Dense(
            units=hp.Int('dense_units', 64, 128, step=32),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_val)
        ),
        keras.layers.Dense(len(actions_list), activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Execução Principal ---
if __name__ == '__main__':
    DATA_PATH = "Libras_Data"
    X, y, actions_list = load_sequential_data(DATA_PATH, augment=True, augment_factor=2)

    encoder = LabelEncoder()
    encoder.fit(actions_list)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    y_encoded = y
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(actions_list))

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='kt_final',
        project_name='libras_lstm',
        overwrite=True
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X, y_categorical, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(1)[0]

    print("\nMelhores hiperparâmetros:")
    for hp_name in ['l2_reg', 'units_lstm_1', 'dropout_1', 'dense_units', 'learning_rate']:
        print(f"{hp_name}: {best_hps.get(hp_name)}")

    final_model = build_model(best_hps)
    history = final_model.fit(X, y_categorical, epochs=80, batch_size=32, validation_split=0.1, callbacks=[stop_early])
    final_model.save("modelo_final_libras.h5")

    # Avaliação final
    y_pred = np.argmax(final_model.predict(X), axis=1)
    cm = confusion_matrix(y_encoded, y_pred)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda')
    plt.legend()
    plt.savefig("training_final.png")
    plt.show()

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions_list, yticklabels=actions_list)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.savefig("confusion_final.png")
    plt.show()

    print("Modelo final salvo como 'modelo_final_libras.h5'")
