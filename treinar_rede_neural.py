import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
NOME_ARQUIVO_DADOS = 'libras_dataset.csv'
df = pd.read_csv(NOME_ARQUIVO_DADOS)

# X são as coordenadas (features), y são as letras (labels)
X = df.drop('letra', axis=1).values
y = df['letra'].values

# --- TRANSFORMAÇÃO DOS RÓTULOS (CRUCIAL PARA REDES NEURAIS) ---
# 1. Converter rótulos de texto ('A', 'B', ..) para números (0, 1, ..)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 2. Converter os rótulos numéricos para o formato "one-hot"
# Ex: 'A' -> 0 -> [1, 0, 0, ..., 0]
# Ex: 'B' -> 1 -> [0, 1, 0, ..., 0]
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Salvar o encoder para usá-lo depois na predição
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("LabelEncoder salvo em 'label_encoder.pkl'")

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# --- CONSTRUÇÃO DO MODELO DA REDE NEURAL ---
model = keras.Sequential([
    # Camada de entrada: o input_shape deve ser o número de colunas em X
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5), # Dropout ajuda a previnir overfitting
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    # Camada de saída: o número de neurônios deve ser o número de letras (classes)
    # A ativação 'softmax' é usada para classificação multiclasse
    keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Loss function para este tipo de problema
              metrics=['accuracy'])

print("\n--- Resumo do Modelo ---")
model.summary()

# --- TREINAMENTO DO MODELO ---
print("\nIniciando o treinamento da Rede Neural...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
print("Treinamento concluído!")

# --- AVALIAÇÃO E SALVAMENTO ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")

# Salvar o modelo treinado no formato do Keras
model.save('libras_model_nn.h5')
print("Modelo da Rede Neural salvo com sucesso em 'libras_model_nn.h5'")