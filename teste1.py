# teste para tentar ver a mão 
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# função para calcular landmarks da mão
def calculate_normalized_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    landmarks_list = []
    
    # Encontra o ponto de referência (pulso - landmark 0)
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    
    for landmark in hand_landmarks.landmark:
        # Normaliza em relação à posição do pulso
        # Isso torna o modelo independente da posição da mão na tela
        normalized_x = landmark.x - base_x
        normalized_y = landmark.y - base_y
        landmarks_list.extend([normalized_x, normalized_y])
        
    return landmarks_list

# incializar o mediapipe e opencv
capture = cv2.VideoCapture(0) # usar a primeira câmera
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# configurar o detector de mãos
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2)

# lista para armazenar os dados
all_data = []
data_columns = []

for i in range(21): # 21 landmarks, cada um com x e y normalizados
    data_columns.extend([f'x{i}', f'y{i}'])
data_columns.insert(0, 'letra') # Adiciona a coluna da letra no início

letra_atual = 'A' # Comece coletando a letra 'A'
print(f"Preparado para coletar dados para a letra: {letra_atual}. Pressione 's' para salvar.")
print("Pressione 'q' para sair. Pressione a tecla da próxima letra para mudar (ex: 'b', 'c'...).")


while capture.isOpened():
    success, image = capture.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    display_image = image.copy()

    # Lógica de Coleta
    key = cv2.waitKey(5) & 0xFF
    
    if key == 27: # 27 é o código da tecla ESC <--- SOLUÇÃO AQUI
        break
    # Se a tecla pressionada for uma letra, muda a letra a ser coletada
    elif 97 <= key <= 122: # Teclas de 'a' a 'z'
        letra_atual = chr(key).upper()
        print(f"Mudou para a letra: {letra_atual}. Pressione 's' para salvar.")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
            
            # Quando 's' é pressionado, salva os landmarks
            if key == ord('s'):
                landmarks_normalizados = calculate_normalized_landmarks(image, hand_landmarks)
                
                # Adiciona a letra atual como a primeira coluna
                linha_de_dados = [letra_atual] + landmarks_normalizados
                all_data.append(linha_de_dados)
                print(f"Salvo! Amostras para '{letra_atual}': {sum(1 for row in all_data if row[0] == letra_atual)}")

    # Mostra a letra atual na tela
    cv2.putText(display_image, f'Coletando: {letra_atual}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Coleta de Dados para Libras', display_image)


capture.release()
cv2.destroyAllWindows()

# Salva todos os dados coletados em um arquivo CSV
if all_data:
    df = pd.DataFrame(all_data, columns=data_columns)
    df.to_csv('libras_dataset.csv', index=False)
    print("Dados salvos com sucesso em 'libras_dataset.csv'")
else:
    print("Nenhum dado foi coletado.")
