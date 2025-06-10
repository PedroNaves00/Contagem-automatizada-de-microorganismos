import os
import cv2
from datetime import datetime

def criar_pasta_resultados(nome_imagem: str) -> str:
    """
    Cria uma pasta para os resultados de uma imagem específica.
    
    Args:
        nome_imagem: Nome do arquivo de imagem original
        
    Returns:
        Caminho da pasta de resultados criada
    """
    # Obtém o diretório do script atual
    diretorio_atual = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Cria pasta resultados se não existir
    pasta_resultados = os.path.join(diretorio_atual, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)
    
    # Cria uma pasta com timestamp para esta execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = os.path.splitext(os.path.basename(nome_imagem))[0]
    pasta_execucao = os.path.join(pasta_resultados, f"{nome_base}_{timestamp}")
    os.makedirs(pasta_execucao, exist_ok=True)
    
    return pasta_execucao

def salvar_imagem(caminho_pasta: str, nome_etapa: str, imagem) -> str:
    """
    Salva uma imagem em uma pasta específica.
    
    Args:
        caminho_pasta: Caminho da pasta onde salvar
        nome_etapa: Nome da etapa do processamento
        imagem: Imagem a ser salva
        
    Returns:
        Caminho completo do arquivo salvo
    """
    # Cria um nome de arquivo seguro
    nome_arquivo = f"{nome_etapa}.png"
    caminho_completo = os.path.join(caminho_pasta, nome_arquivo)
    
    # Salva a imagem
    cv2.imwrite(caminho_completo, imagem)
    
    return caminho_completo 