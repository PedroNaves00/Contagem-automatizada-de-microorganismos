import cv2
import os
from src.pre_processamento import PipelinePreProcessamento
from src.segmentacao import PipelineSegmentacao
from src.utils import criar_pasta_resultados

def obter_caminho_imagem(nome_arquivo: str) -> str:
  
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    diretorio_pai = os.path.dirname(diretorio_atual)
    caminho_imagens = os.path.join(diretorio_pai, "imagens_teste")
    return os.path.join(caminho_imagens, nome_arquivo)

def executar_pipeline(caminho_imagem: str, mostrar_resultados: bool = True):
    # Carregar a imagem original
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em: {caminho_imagem}")
    
    # Criar pasta para resultados
    pasta_resultados = criar_pasta_resultados(caminho_imagem)
    
    # Pré-processamento
    pre_processador = PipelinePreProcessamento(
        kernel_size=(5, 5),
        clahe_clip_limit=2.0,
        clahe_grid_size=(8, 8),
        threshold_value=127
    )
    img_processada = pre_processador.executar_pipeline(img, mostrar_resultados, pasta_resultados)
    
    # Segmentação
    segmentador = PipelineSegmentacao(
        min_area=100,
        max_area=1000
    )
    img_segmentada, contornos = segmentador.executar_pipeline(img_processada, mostrar_resultados, pasta_resultados)
    
    # Exibir resultado final
    if mostrar_resultados:
        cv2.imshow("Resultado Final", img_segmentada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img_segmentada, contornos

if __name__ == "__main__":
    # Chamar o pipeline usando caminho relativo
    nome_arquivo = "imagemRGB-1.jpg"
    caminho_imagem = obter_caminho_imagem(nome_arquivo)
    executar_pipeline(caminho_imagem)
