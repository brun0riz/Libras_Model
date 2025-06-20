import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from collections import deque

# --- PARÂMETROS E CONFIGURAÇÕES ---
MODEL_PATH = 'libras_model_lstm_final.h5'
ENCODER_PATH = 'label_encoder.pkl'

# --- PARÂMETROS DE USABILIDADE (AJUSTE CONFORME NECESSÁRIO) ---
CONFIDENCE_THRESHOLD = 0.70
FRAMES_TO_CONFIRM = 10
SEQUENCE_LENGTH = 30

# --- CORES PARA VISUALIZAÇÃO ---
COR_TEXTO = (255, 255, 255)
COR_PREDICAO_BOA = (0, 255, 0) # Verde para predições acima do limiar

# --- CARREGAR MODELO E ENCODER ---
try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print("Modelo e Encoder carregados com sucesso.")
    CLASSES = encoder.classes_
    print(f"Classes que o modelo reconhece: {CLASSES}")
except Exception as e:
    print(f"Erro crítico ao carregar modelo ou encoder: {e}")
    exit()

# Função para normalizar landmarks
def calculate_normalized_landmarks_optimized(hand_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    base_point = coords[0]
    relative_coords = coords - base_point
    max_dist = np.max(np.linalg.norm(relative_coords, axis=1))
    if max_dist == 0: return np.zeros(21 * 2)
    normalized_coords = relative_coords / max_dist
    return normalized_coords.flatten()

# --- INICIALIZAÇÃO MEDIAPIPE E OPENCV ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
cap = cv2.VideoCapture(0)

# --- VARIÁVEIS PARA LÓGICA CONTÍNUA ---
sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence = []
last_stable_prediction = ""
prediction_buffer = deque(maxlen=FRAMES_TO_CONFIRM)

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    display_image = image.copy()
    
    current_prediction_label = ""
    current_confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = calculate_normalized_landmarks_optimized(hand_landmarks)
        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(list(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            
            current_confidence = np.max(prediction)
            predicted_index = np.argmax(prediction)
            
            if current_confidence >= CONFIDENCE_THRESHOLD:
                current_prediction_label = CLASSES[predicted_index]
                prediction_buffer.append(current_prediction_label)

                if len(prediction_buffer) == FRAMES_TO_CONFIRM and len(set(prediction_buffer)) == 1:
                    if current_prediction_label != last_stable_prediction and current_prediction_label != 'fundo':
                        sentence.append(current_prediction_label)
                        last_stable_prediction = current_prediction_label
            else:
                prediction_buffer.clear()
    else:
        prediction_buffer.clear()

    # --- LÓGICA DE VISUALIZAÇÃO NA TELA ---
    if current_confidence >= CONFIDENCE_THRESHOLD:
        text = f"{current_prediction_label} ({current_confidence:.2f})"
        cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COR_PREDICAO_BOA, 2)

    cv2.rectangle(display_image, (10, 400), (630, 470), (0, 0, 0), -1)
    cv2.putText(display_image, ' '.join(sentence), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, COR_TEXTO, 3)

    cv2.imshow('Reconhecimento Contínuo de Libras', display_image)
    
    # --- GERENCIAMENTO DE TECLAS ---
    key = cv2.waitKey(5) & 0xFF
    if key == 27: # Tecla ESC para sair
        break
    elif key == 8: # Tecla BACKSPACE para apagar a última letra
        if sentence:
            sentence.pop()
        # Reseta a última predição estável para permitir que a mesma letra seja adicionada novamente
        last_stable_prediction = "" 
            
    elif key == 32: # Tecla ESPAÇO para limpar a frase inteira
        sentence = []
        last_stable_prediction = ""
            
cap.release()
cv2.destroyAllWindows()