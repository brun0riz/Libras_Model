import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import kerastuner as kt

def load_sequential_data(data_path):
    """
    Carrega os dados sequenciais salvos em arquivos .npy.
    Espera uma estrutura de pastas: data_path/action_name/sequence_num.npy
    """
    sequences, labels = [], [] # Lista para armazenar os gestos e rótulos
    actions = [action for action in os.listdir(data_path) if not action.startswith('.')]
    
    label_map = {label: num for num, label in enumerate(actions)} # Mapeia ações para números

    print("Carregando dados...")
    for action in actions:
        action_path = os.path.join(data_path, action)
        sequence_files = os.listdir(action_path)
        for seq_file in sequence_files:
            res = np.load(os.path.join(action_path, seq_file))
            sequences.append(res)
            labels.append(label_map[action])
    
    print(f"Dados carregados. Total de {len(sequences)} sequências.")
    return np.array(sequences), np.array(labels), actions

# --- 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
DATA_PATH = "Libras_Data"
X, y, actions = load_sequential_data(DATA_PATH)

# Codificar os rótulos (letras) para números (embora já tenhamos feito, o encoder é útil)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform([actions[i] for i in y])

# Salvar o encoder para usá-lo depois na predição
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("LabelEncoder salvo em 'label_encoder.pkl'")

y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Dividir os dados em treino e teste para a busca de hiperparâmetros
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# --- 3. FUNÇÃO PARA CRIAR O HIPERMODELO (LSTM) ---
def build_model(hp):
    """
    Constrói um modelo LSTM que o Keras Tuner pode otimizar.
    """
    model = keras.Sequential()
    
    # Camada LSTM com número de unidades a ser otimizado
    model.add(keras.layers.LSTM(
        units=hp.Int('units_lstm', min_value=32, max_value=128, step=32),
        return_sequences=True, # Importante quando há LSTMs empilhadas
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(keras.layers.LSTM(
        units=hp.Int('units_lstm_2', min_value=32, max_value=128, step=32),
        return_sequences=False # A última camada LSTM não retorna sequência
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Camada Densa de saída
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

    # Compilação do modelo
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 4. BUSCA DE HIPERPARÂMETROS COM KERAS TUNER ---
print("\nIniciando a busca por hiperparâmetros...")

# Usaremos o Hyperband, um algoritmo de busca eficiente
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30, # Máximo de épocas para os melhores modelos
    factor=3,
    directory='keras_tuner_dir',
    project_name='libras_lstm'
)

# Callbacks para parar o treino mais cedo se não houver melhora
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

# Pega os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n--- Melhores Hiperparâmetros Encontrados ---")
print(f"Unidades LSTM (camada 1): {best_hps.get('units_lstm')}")
print(f"Dropout (camada 1): {best_hps.get('dropout_1'):.2f}")
print(f"Unidades LSTM (camada 2): {best_hps.get('units_lstm_2')}")
print(f"Dropout (camada 2): {best_hps.get('dropout_2'):.2f}")
print(f"Taxa de Aprendizagem: {best_hps.get('learning_rate')}")
print("-" * 50)


# --- 5. TREINAMENTO FINAL COM OS MELHORES HIPERPARÂMETROS ---
print("\n--- Treinamento Final com Todos os Dados e Melhores Parâmetros ---")

# Constrói o modelo com os melhores hiperparâmetros encontrados
final_model = tuner.hypermodel.build(best_hps)

# Treina o modelo final com todos os dados
history = final_model.fit(X, y_categorical, epochs=100, batch_size=32, validation_split=0.1, callbacks=[stop_early])

# Salva o modelo final
final_model.save('libras_model_lstm_tuned.h5')
print("\nModelo final salvo com sucesso em 'libras_model_lstm_tuned.h5'")