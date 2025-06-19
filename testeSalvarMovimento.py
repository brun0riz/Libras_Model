import cv2
import mediapipe as mp
import numpy as np
import os


# Caminho para a pasta onde os dados serão salvos
DATA_PATH = "Libras_Data"

# Lista para as letras/ações que serão coletadas
actions = np.array(['A', 'B', 'C','Ç', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Diconário para mapear teclas do teclado para as letras/ações
# Ç agora é acionada pela tecla '1'
key_map = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 
    'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 
    'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 
    'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 
    'Y': 'Y', 'Z': 'Z',
    '1': 'Ç'  # A tecla '1' agora seleciona o gesto 'Ç'
}

# Número de frames que compõem uma única sequência de gesto
SEQUENCE_LENGTH = 30

# Parâmetros para a inicialização do MediaPipe Hands
hands_params = {
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5
}


def calculate_normalized_landmarks_optimized(hand_landmarks):
    """
    Calcula os landmarks normalizados para serem invariantes à translação e à escala.
    - O punho (landmark 0) se torna a origem (0,0).
    - A escala é normalizada pela distância máxima de qualquer ponto da mão ao punho.
    """
    # Obter todas as coordenadas e o ponto de referência (punho)
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x, landmark.y])
    
    # Converter para um array NumPy para facilitar os cálculos
    coords = np.array(coords)
    
    base_point = coords[0] # Landmark 0 é o punho
    relative_coords = coords - base_point # Calcular coordenadas relativas ao punho
    
    max_dist = np.max(np.linalg.norm(relative_coords, axis=1)) # Distância máxima de qualquer ponto ao punho
    
    if max_dist == 0:
        return np.zeros_like(relative_coords).flatten()

    normalized_coords = relative_coords / max_dist
    
    # Achatar a lista de coordenadas para salvar no formato (x0, y0, x1, y1, ...)
    return normalized_coords.flatten()


def main():
    """
    Função principal que abre a câmera e gerencia a coleta de dados.
    """
    # Cria as pastas para cada ação, se não existirem
    for action in actions:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    # Inicializa o MediaPipe Hands e a captura de vídeo
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(**hands_params)
    capture = cv2.VideoCapture(0)

    letra_atual = actions[0] # Letra atual para coleta
    recording = False # Estado de gravação
    sequence_data = [] # Lista temporária para armazenar os dados da sequência

    print("--- Coleta de Dados para Libras ---")
    print(f"Letras a serem coletadas: {', '.join(actions)}")
    print("Pressione a letra no teclado para selecionar (ou '1' para 'Ç').") # <<-- Instrução atualizada
    print("Pressione 'ESPAÇO' para gravar uma sequência.")
    print("Pressione 'ESC' para sair.")
    
    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("Ignorando frame vazio da câmera.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        display_image = image.copy()

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        
        key_char = chr(key).upper()

        # Verifica se a tecla pressionada corresponde a uma ação
        if key_char in key_map:
            letra_atual = key_map[key_char] # Pega a etiqueta correta (ex: 'Ç' quando a tecla é '1')
            print(f"\nSelecionado: Gesto '{letra_atual}' (acionado pela Tecla: '{key_char}')")
            recording = False
            sequence_data = []
        
        if key == 32:
            if not recording:
                print(f"Iniciando gravação para '{letra_atual}'...")
                recording = True
                sequence_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if recording:
            # O texto na tela agora mostrará 'Ç' corretamente porque a variável 'letra_atual' contém o caractere certo
            cv2.putText(display_image, f'GRAVANDO: {letra_atual} - Frame {len(sequence_data)}/{SEQUENCE_LENGTH}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            if results.multi_hand_landmarks:
                landmarks = calculate_normalized_landmarks_optimized(results.multi_hand_landmarks[0])
                sequence_data.append(landmarks)
            else:
                sequence_data.append(np.zeros(21 * 2))

            if len(sequence_data) == SEQUENCE_LENGTH:
                sequence_number = len(os.listdir(os.path.join(DATA_PATH, letra_atual)))
                save_path = os.path.join(DATA_PATH, letra_atual, f"{sequence_number}.npy")
                
                np.save(save_path, np.array(sequence_data))
                print(f"Sequência para '{letra_atual}' salva com sucesso em {save_path}")
                
                recording = False
                sequence_data = []
        else:
             # O texto na tela também mostrará 'Ç' aqui
             cv2.putText(display_image, f"Aponte para: '{letra_atual}'", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
             cv2.putText(display_image, "Pressione ESPACO para gravar", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Coleta de Dados para Libras', display_image)

    capture.release()
    cv2.destroyAllWindows()
    print("\nColeta de dados finalizada.")


if __name__ == '__main__':
    main()