# ver_modelo.py (versão final para inspeção completa)
import tensorflow as tf
from tensorflow import keras

MODEL_NAME = 'libras_model_lstm_cv_tuned.h5'
print(f"Carregando o modelo '{MODEL_NAME}'...")

try:
    model = keras.models.load_model(MODEL_NAME)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# 1. Resumo da Arquitetura
print("\n--- Resumo da Arquitetura do Modelo ---")
model.summary()

# 2. Contagem de Camadas
num_camadas = len(model.layers)
print("\n" + "="*50)
print(f"✅ Total de Camadas: {num_camadas}")
print("="*50)

# 3. Listando as Funções de Ativação de Cada Camada
print("\n--- Funções de Ativação por Camada ---")
for camada in model.layers:
    # Acessa a configuração da camada, que é um dicionário
    config = camada.get_config()
    
    # Pega a função de ativação. Se não houver (ex: Dropout), retorna 'N/A'
    ativacao = config.get('activation', 'N/A')
    
    # Para LSTMs, as ativações principais são internas e fixas
    if isinstance(camada, keras.layers.LSTM):
        ativacao_recorrente = config.get('recurrent_activation') # Sigmoid para os portões
        ativacao_principal = config.get('activation')          # Tanh para o conteúdo
        print(f"- Camada: {camada.name} (LSTM)")
        print(f"  - Ativação Principal (conteúdo): {ativacao_principal} (padrão)")
        print(f"  - Ativação Recorrente (portões): {ativacao_recorrente} (padrão)")
    else:
        print(f"- Camada: {camada.name} ({camada.__class__.__name__}) -> Ativação: {ativacao}")