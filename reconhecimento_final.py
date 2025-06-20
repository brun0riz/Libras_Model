import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# --- PARÂMETROS E CONFIGURAÇÕES ---
MODEL_PATH = 'libras_model_lstm_final.h5'
ENCODER_PATH = 'label_encoder.pkl'

# Confiança mínima para uma predição ser considerada válida
CONFIDENCE_THRESHOLD = 0.85 
# Frames para capturar para uma predição
SEQUENCE_LENGTH = 30 
# Frames que a mão deve estar parada antes de iniciar a captura
FRAMES_TO_BE_STABLE = 10 

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

# Função para normalizar landmarks (deve ser idêntica à da coleta)
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

# --- VARIÁVEIS DE ESTADO PARA A LÓGICA DE GATILHO ---
# Estados: 'WAITING', 'STABILIZING', 'CAPTURING', 'PREDICTING'
current_state = 'WAITING'
sequence_data = []
sentence = []
last_prediction = ""
stable_counter = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    display_image = image.copy()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- MÁQUINA DE ESTADOS ---
        if current_state == 'WAITING':
            current_state = 'STABILIZING'
            stable_counter = 0

        elif current_state == 'STABILIZING':
            cv2.putText(display_image, "Estabilizando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            stable_counter += 1
            if stable_counter >= FRAMES_TO_BE_STABLE:
                current_state = 'CAPTURING'
                sequence_data = []
                print("Mão estabilizada. Iniciando captura...")

        elif current_state == 'CAPTURING':
            cv2.putText(display_image, f"Gravando... {len(sequence_data)}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            landmarks = calculate_normalized_landmarks_optimized(hand_landmarks)
            if not np.all(landmarks == 0):
                sequence_data.append(landmarks)
            
            if len(sequence_data) == SEQUENCE_LENGTH:
                current_state = 'PREDICTING'

        elif current_state == 'PREDICTING':
            print("Captura completa. Realizando predição...")
            input_data = np.expand_dims(sequence_data, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            
            confidence = np.max(prediction)
            predicted_index = np.argmax(prediction)
            predicted_label = CLASSES[predicted_index]
            
            # Verifica se a predição é confiável e não é a classe 'fundo'
            if confidence >= CONFIDENCE_THRESHOLD and predicted_label != 'fundo':
                last_prediction = f"{predicted_label} ({confidence:.2f})"
                if not sentence or sentence[-1] != predicted_label:
                    sentence.append(predicted_label)
            else:
                last_prediction = f"Incerto ou Fundo ({confidence:.2f})"
            
            print(f"Predição: {last_prediction}")
            current_state = 'WAITING' # Reseta para o início do ciclo

    else: # Se a mão não for detectada
        if current_state != 'WAITING':
            print("Mão perdida. Resetando estado.")
        current_state = 'WAITING'
        stable_counter = 0
        sequence_data = []

    # --- EXIBIÇÃO NA TELA ---
    # Mostra a última predição válida
    if last_prediction:
        cv2.putText(display_image, last_prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra a sentença formada
    cv2.rectangle(display_image, (10, 400), (630, 470), (0, 0, 0), -1)
    cv2.putText(display_image, ' '.join(sentence), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow('Reconhecimento de Libras', display_image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27: break # ESC para sair
    if key == 32: sentence = [] # ESPAÇO para limpar

cap.release()
cv2.destroyAllWindows()