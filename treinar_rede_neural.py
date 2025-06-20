import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pickle
import keras_tuner as kt
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. FUNÇÃO DE CARREGAMENTO DE DADOS (robusta) ---
def load_sequential_data(data_path):
    sequences, labels = [], []
    # Garante que a ordem das ações seja consistente
    actions = sorted([action for action in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, action)) and not action.startswith('.')])
    label_map = {label: num for num, label in enumerate(actions)}
    
    print("Carregando dados...")
    print(f"Classes encontradas: {actions}")
    
    for action in actions:
        action_path = os.path.join(data_path, action)
        for seq_file in os.listdir(action_path):
            if seq_file.endswith('.npy'):
                filepath = os.path.join(action_path, seq_file)
                try:
                    res = np.load(filepath)
                    # Verifica se a sequência tem o shape esperado (30, 42)
                    if res.shape == (30, 42): 
                        sequences.append(res)
                        labels.append(label_map[action])
                    else:
                        print(f"Aviso: Arquivo '{filepath}' ignorado. Formato {res.shape} inválido.")
                except Exception as e:
                    print(f"Aviso: Erro ao carregar o arquivo '{filepath}': {e}. Ignorando.")

    print(f"Dados carregados. Total de {len(sequences)} sequências válidas.")
    if not sequences:
        raise ValueError("Nenhum dado válido foi carregado.")
    return np.array(sequences), np.array(labels), actions

# --- 2. FUNÇÃO PARA CRIAR O HIPERMODELO ---
def build_model(hp):
    model = keras.Sequential([
        keras.Input(shape=(30, 42)), # Shape da entrada: 30 frames, 42 landmarks
        keras.layers.LSTM(
            units=hp.Int('units_lstm_1', min_value=32, max_value=128, step=32),
            return_sequences=True
        ),
        keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)),
        keras.layers.LSTM(
            units=hp.Int('units_lstm_2', min_value=32, max_value=128, step=32),
            return_sequences=False
        ),
        keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
        keras.layers.Dense(y_categorical.shape[1], activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Início da Execução Principal ---
if __name__ == '__main__':
    # ETAPA DE PREPARAÇÃO
    DATA_PATH = "Libras_Data"
    X, y, actions_list = load_sequential_data(DATA_PATH)
    
    # Salvar o LabelEncoder que mapeia nomes de classes para números
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y) # y já vem como números, mas fit_transform é mais seguro
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print(f"LabelEncoder salvo em 'label_encoder.pkl'. Classes: {encoder.classes_}")

    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    # ETAPA DE BUSCA DE HIPERPARÂMETROS
    print("\nIniciando a busca por hiperparâmetros...")
    # Usando Hyperband, um método eficiente para busca
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='keras_tuner_dir',
        project_name='libras_lstm',
        overwrite=True
    )
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    # O tuner usa sua própria divisão de validação, então não precisamos de K-Fold aqui.
    # K-Fold foi usado na versão anterior para avaliar os trials, mas a busca interna do tuner é mais direta.
    tuner.search(X, y_categorical, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n" + "="*50)
    print("      MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print("="*50)
    for hp in ['units_lstm_1', 'dropout_1', 'units_lstm_2', 'dropout_2', 'learning_rate']:
        print(f"{hp}: {best_hps.get(hp)}")
    print("="*50)

    # ETAPA DE TREINAMENTO FINAL com todos os dados
    print("\n--- Treinamento Final com Todos os Dados e Melhores Parâmetros ---")
    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(X, y_categorical, epochs=100, batch_size=32, validation_split=0.1, callbacks=[stop_early])
    
    model_filename = 'libras_model_lstm_final.h5'
    final_model.save(model_filename)
    print(f"\nModelo final salvo com sucesso em '{model_filename}'")
    
    # ETAPA DE ANÁLISE PÓS-TREINAMENTO
    print("\n--- Gerando gráficos de análise ---")

    # 1. Gráficos de Acurácia e Perda
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Histórico de Acurácia')
    plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Histórico de Perda')
    plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
    plt.savefig('training_history.png')
    plt.show()

    # 2. Matriz de Confusão
    y_pred = final_model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_categorical, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions_list, yticklabels=actions_list)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    print("Gráficos de análise salvos em 'training_history.png' e 'confusion_matrix.png'")