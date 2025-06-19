import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pickle
import kerastuner as kt

def load_sequential_data(data_path):
    """
    Carrega os dados sequenciais salvos em arquivos .npy.
    Espera uma estrutura de pastas: data_path/action_name/sequence_num.npy
    """
    sequences, labels = [], []
    # Lista todas as subpastas (que são os nomes das ações/letras)
    actions = [action for action in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, action)) and not action.startswith('.')]
    
    # Cria um mapa de texto para número (ex: 'A' -> 0, 'B' -> 1, ...)
    label_map = {label: num for num, label in enumerate(actions)}

    print("Carregando dados...")
    for action in actions:
        action_path = os.path.join(data_path, action)
        sequence_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        for seq_file in sequence_files:
            res = np.load(os.path.join(action_path, seq_file))
            sequences.append(res)
            labels.append(label_map[action])
    
    print(f"Dados carregados. Total de {len(sequences)} sequências para {len(actions)} ações.")
    return np.array(sequences), np.array(labels), actions

# --- 2. FUNÇÃO PARA CRIAR O HIPERMODELO (LSTM) ---
def build_model(hp):
    """
    Constrói um modelo LSTM que o Keras Tuner pode otimizar.
    'hp' é um objeto de hiperparâmetros que o Tuner passa para a função.
    """
    model = keras.Sequential()
    
    # Camada LSTM com número de unidades a ser otimizado
    model.add(keras.layers.LSTM(
        units=hp.Int('units_lstm_1', min_value=32, max_value=128, step=32),
        return_sequences=True, # Necessário para empilhar camadas LSTM
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    # Segunda camada LSTM
    model.add(keras.layers.LSTM(
        units=hp.Int('units_lstm_2', min_value=32, max_value=128, step=32),
        return_sequences=False # A última camada LSTM não retorna sequência
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Camada Densa de saída
    model.add(keras.layers.Dense(y_categorical.shape[1], activation='softmax'))

    # Compilação do modelo com taxa de aprendizado otimizável
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Início da Execução Principal ---
if __name__ == '__main__':
    # --- ETAPA DE PREPARAÇÃO ---
    DATA_PATH = "Libras_Data"
    X, y, actions_list = load_sequential_data(DATA_PATH)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform([actions_list[i] for i in y])

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("LabelEncoder salvo em 'label_encoder.pkl'")

    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    # --- ETAPA DE BUSCA DE HIPERPARÂMETROS COM VALIDAÇÃO CRUZADA ---
    print("\nIniciando a busca por hiperparâmetros com Validação Cruzada K-Fold...")

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='keras_tuner_cv_dir',
        project_name='libras_lstm_cv'
    )

    N_SPLITS = 5
    NUM_TRIALS = 20
    EPOCHS_PER_FOLD = 50

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for trial_num in range(NUM_TRIALS):
        trial = tuner.oracle.create_trial(tuner.tuner_id)
        hps = trial.hyperparameters
        print(f"\n--- Iniciando Trial {trial.trial_id} / {NUM_TRIALS} ---")
        print(f"Hiperparâmetros: {hps.get_config()['values']}")

        fold_accuracies = []
        for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded), 1):
            print(f"  > Processando Fold {fold}/{N_SPLITS}...")
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_categorical[train_index], y_categorical[val_index]
            
            model = tuner.hypermodel.build(hps)
            model.fit(X_train, y_train, epochs=EPOCHS_PER_FOLD, validation_data=(X_val, y_val), 
                      callbacks=[stop_early], verbose=0)
            
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_accuracies.append(accuracy)

        average_accuracy = np.mean(fold_accuracies)
        print(f"  > Acurácia Média para o Trial {trial.trial_id}: {average_accuracy:.4f}")
        
        tuner.oracle.update_trial(trial.trial_id, {'val_accuracy': average_accuracy})
        tuner.oracle.end_trial(trial.trial_id)

    # --- ETAPA DE TREINAMENTO FINAL ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n" + "="*50)
    print("      MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print("="*50)
    print(f"Unidades LSTM (camada 1): {best_hps.get('units_lstm_1')}")
    print(f"Dropout (camada 1): {best_hps.get('dropout_1'):.2f}")
    print(f"Unidades LSTM (camada 2): {best_hps.get('units_lstm_2')}")
    print(f"Dropout (camada 2): {best_hps.get('dropout_2'):.2f}")
    print(f"Taxa de Aprendizagem: {best_hps.get('learning_rate')}")
    print("="*50)

    print("\n--- Treinamento Final com Todos os Dados e Melhores Parâmetros ---")

    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(X, y_categorical, epochs=100, batch_size=32, validation_split=0.1, callbacks=[stop_early])

    final_model.save('libras_model_lstm_cv_tuned.h5')
    print("\nModelo final salvo com sucesso em 'libras_model_lstm_cv_tuned.h5'")