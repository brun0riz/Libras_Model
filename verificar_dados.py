# listar_arquivos_para_deletar.py
import os
import numpy as np
from collections import defaultdict

DATA_PATH = "Libras_Data"
# Dicionário para armazenar os arquivos agrupados por seu formato
shapes = defaultdict(list)

print(f"Verificando formatos (shapes) dos arquivos em '{DATA_PATH}'...")

# 1. Coleta os formatos de todos os arquivos
for root, dirs, files in os.walk(DATA_PATH):
    for name in files:
        if name.endswith(".npy"):
            filepath = os.path.join(root, name)
            try:
                data = np.load(filepath)
                shapes[data.shape].append(filepath)
            except Exception as e:
                print(f"Não foi possível ler o arquivo {filepath}: {e}")

# 2. Encontra o formato mais comum (o formato correto)
if not shapes:
    print("Nenhum arquivo de dados encontrado.")
    exit()

# Encontra o shape que tem a maior contagem de arquivos
correct_shape = max(shapes, key=lambda k: len(shapes[k]))
print(f"O formato padrão (correto) foi identificado como: {correct_shape} com {len(shapes[correct_shape])} arquivos.")

# 3. Lista todos os arquivos que NÃO têm o formato correto
print("\n--- Lista de arquivos a serem DELETADOS ---")
arquivos_para_deletar = []
for shape, file_list in shapes.items():
    if shape != correct_shape:
        print(f"\nArquivos com o formato incorreto {shape}:")
        for filepath in file_list:
            print(f"  - {filepath}")
            arquivos_para_deletar.append(filepath)

if not arquivos_para_deletar:
    print("Nenhum arquivo com formato incompatível foi encontrado. Seu dataset está limpo!")
else:
    print(f"\nTotal de {len(arquivos_para_deletar)} arquivos a serem removidos.")