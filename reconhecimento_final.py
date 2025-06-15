import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# --- CARREGAR MODELO E ENCODER ---
try:
    model = load_model('libras_model_nn.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar os arquivos do modelo: {e}")
    print("Execute o script 'treinar_rede_neural.py' primeiro.")
    exit()

# O resto do código é muito parecido com o anterior
# Função para calcular landmarks normalizados
def calculate_normalized_landmarks(image, hand_landmarks):
    # ... (código idêntico ao script anterior)
    h, w, _ = image.shape
    landmarks_list = []
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    for landmark in hand_landmarks.landmark:
        normalized_x = landmark.x - base_x
        normalized_y = landmark.y - base_y
        landmarks_list.extend([normalized_x, normalized_y])
    return landmarks_list

# Inicializa MediaPipe e OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
cap = cv2.VideoCapture(0)

# Variáveis para exibir a letra e construir o nome
nome = ""
limite_confianca = 0.9
contador_frames = 0
letra_estavel = ""

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    display_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks_normalizados = calculate_normalized_landmarks(image, hand_landmarks)
            dados_para_predicao = np.array(landmarks_normalizados).reshape(1, -1)
            
            # --- PREDIÇÃO COM O MODELO KERAS ---
            predicao_prob = model.predict(dados_para_predicao)[0]
            confianca = np.max(predicao_prob)
            
            # Pega o índice da maior probabilidade
            indice_predito = np.argmax(predicao_prob)
            # Converte o índice de volta para a letra original
            letra_predita = encoder.inverse_transform([indice_predito])[0]

            if confianca > limite_confianca:
                if letra_predita == letra_estavel:
                    contador_frames += 1
                else:
                    letra_estavel = letra_predita
                    contador_frames = 0
                
                if contador_frames > 20:
                    if len(nome) == 0 or nome[-1] != letra_estavel:
                        nome += letra_estavel
                    contador_frames = 0
                
                cv2.putText(display_image, f"{letra_predita} ({confianca:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        letra_estavel = ""
        contador_frames = 0

    cv2.rectangle(display_image, (10, 400), (630, 470), (0, 0, 0), -1)
    cv2.putText(display_image, f"Nome: {nome}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow('Reconhecimento com Rede Neural', display_image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27: break
    elif key == 8: nome = nome[:-1]

cap.release()
cv2.destroyAllWindows()