import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# Função para calcular os landmarks normalizados (sem alterações aqui)
def calculate_normalized_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    landmarks_list = []
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    for landmark in hand_landmarks.landmark:
        normalized_x = landmark.x - base_x
        normalized_y = landmark.y - base_y
        landmarks_list.extend([normalized_x, normalized_y])
    return landmarks_list

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)

# Inicia a captura de vídeo
capture = cv2.VideoCapture(0)

# Lista para armazenar todos os dados coletados
all_data = []
data_columns = []
for i in range(21):
    data_columns.extend([f'x{i}', f'y{i}'])
data_columns.insert(0, 'letra')

# --- CONFIGURAÇÃO DA COLETA ---
letra_atual = 'A'
print(f"Preparado para coletar dados para a letra: {letra_atual}.")
print("Pressione a BARRA DE ESPAÇO para salvar a amostra.")
print("Pressione a tecla da próxima letra para mudar (ex: 'b', 'c'...).")
print("Pressione 'ESC' para sair.")


while capture.isOpened():
    success, image = capture.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    display_image = image.copy()

    key = cv2.waitKey(5) & 0xFF
    
    # Tecla ESC para sair
    if key == 27:
        break
        
    # Teclas de 'a' a 'z' para selecionar a letra
    elif 97 <= key <= 122:
        letra_atual = chr(key).upper()
        print(f"Mudou para a letra: {letra_atual}. Pressione ESPAÇO para salvar.")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # MUDANÇA AQUI: Trocamos ord('s') por 32 (barra de espaço) para salvar
            if key == 32:
                landmarks_normalizados = calculate_normalized_landmarks(image, hand_landmarks)
                linha_de_dados = [letra_atual] + landmarks_normalizados
                all_data.append(linha_de_dados)
                print(f"Salvo! Amostras para '{letra_atual}': {sum(1 for row in all_data if row[0] == letra_atual)}")

    # Mostra as instruções na tela
    cv2.putText(display_image, f'Coletando: {letra_atual}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display_image, "Pressione ESPACO para salvar", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display_image, "Pressione a letra para mudar", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display_image, "Pressione ESC para sair", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Coleta de Dados para Libras', display_image)

capture.release()
cv2.destroyAllWindows()

# Salva os dados no arquivo CSV
if all_data:
    df = pd.DataFrame(all_data, columns=data_columns)
    df.to_csv('libras_dataset.csv', index=False)
    print("Dados salvos com sucesso em 'libras_dataset.csv'")
else:
    print("Nenhum dado foi coletado.")