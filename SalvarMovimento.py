import cv2
import mediapipe as mp
import numpy as np
import os

# pasta de letras
DATA_PATH = "Libras_Data"

# Lista para as letras/ações que serão coletadas
actions = np.array(['A', 'B', 'C','cedinha_a_cecedilha', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'fundo'])
# cedinha_a_cecedilha é a letra 'Ç' em Libras, representada como 'cedinha_a_cecedilha'
# fundo é a classe para movimentos aleatórios ou sem gesto específico

# Dicionário para mapear teclas para ações
key_map = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 
    'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 
    'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 
    'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 
    'Y': 'Y', 'Z': 'Z',
    '1': 'cedinha_a_cecedilha',
    '0': 'fundo'  # Tecla '0' para a classe 'fundo'
}

# Parâmetros de coleta
SEQUENCE_LENGTH = 30 # Frames por sequência
hands_params = {
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5
}

def calculate_normalized_landmarks_optimized(hand_landmarks):
    """Normaliza os landmarks da mão para serem invariantes à posição e ao tamanho."""
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    base_point = coords[0]  # Usa o pulso como ponto de referência
    relative_coords = coords - base_point
    max_dist = np.max(np.linalg.norm(relative_coords, axis=1))
    
    # Proteção contra divisão por zero
    if max_dist == 0:
        return np.zeros(21 * 2)
        
    normalized_coords = relative_coords / max_dist
    return normalized_coords.flatten()


def main():
    for action in actions:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(**hands_params)
    capture = cv2.VideoCapture(0)

    letra_atual = actions[0]
    recording = False
    sequence_data = []

    print("--- Coleta de Dados para Libras ---")
    print(f"Letras a serem coletadas: {', '.join(actions)}")
    print("Pressione a letra no teclado para selecionar (ou '1' para 'Ç', '0' para 'fundo').")
    print("Para a classe 'fundo', faça movimentos aleatórios com a mão, descanse-a, etc.")
    print("Pressione 'ESPAÇO' para iniciar a gravação (após uma pausa de 3s).")
    print("Pressione 'ESC' para sair.")
    
    while capture.isOpened():
        success, image = capture.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        display_image = image.copy()

        key = cv2.waitKey(5) & 0xFF
        if key == 27: break
        
        # Lógica para selecionar a letra
        if not recording:
            key_char = chr(key).upper()
            if key_char in key_map:
                letra_atual = key_map[key_char]
                print(f"\nSelecionado: Gesto '{letra_atual}' (acionado pela Tecla: '{key_char}')")
        
        # Lógica de gravação com contagem regressiva
        if key == 32 and not recording: # Tecla Espaço
            
            recording = True
            sequence_data = []

        # Lógica durante a gravação
        if recording:
            if results.multi_hand_landmarks:
                landmarks = calculate_normalized_landmarks_optimized(results.multi_hand_landmarks[0])
                
                # CORREÇÃO CRÍTICA: Verifica se os landmarks são válidos
                if np.all(landmarks == 0):
                    print("ERRO DE LANDMARK! Frame inválido (todos zeros). Gravação cancelada.")
                    recording = False
                else:
                    sequence_data.append(landmarks)
                    cv2.putText(display_image, f'GRAVANDO: {letra_atual} - {len(sequence_data)}/{SEQUENCE_LENGTH}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("MÃO NÃO DETECTADA! Gravação cancelada.")
                recording = False

            # Se a sequência está completa ou foi cancelada
            if len(sequence_data) == SEQUENCE_LENGTH:
                sequence_number = len(os.listdir(os.path.join(DATA_PATH, letra_atual)))
                save_path = os.path.join(DATA_PATH, letra_atual, f"{sequence_number}.npy")
                np.save(save_path, np.array(sequence_data))
                print(f"Sequência para '{letra_atual}' salva como '{sequence_number}.npy'")
                recording = False
        
        else: # Se não está gravando
            cv2.putText(display_image, f"Aponte para: '{letra_atual}'", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, "Pressione ESPACO para gravar", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Desenha os landmarks na tela sempre que detectados
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Coleta de Dados para Libras', display_image)

    capture.release()
    cv2.destroyAllWindows()
    print("\nColeta de dados finalizada.")

if __name__ == '__main__':
    main()