import cv2
import os
from src.pre_processamento import PipelinePreProcessamento
from src.segmentacao import PipelineSegmentacao
from src.contagem import OperacoesContagem  # <-- Importar
from src.utils import criar_pasta_resultados, salvar_imagem # <-- Importar salvar_imagem

def obter_caminho_imagem(nome_arquivo: str) -> str:
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    diretorio_pai = os.path.dirname(diretorio_atual)
    caminho_imagens = os.path.join(diretorio_pai, "imagens_teste")
    return os.path.join(caminho_imagens, nome_arquivo)

def executar_pipeline(caminho_imagem: str, mostrar_resultados: bool = True):
    # Carregar a imagem original
    img_original = cv2.imread(caminho_imagem)
    if img_original is None:
        raise FileNotFoundError(f"Imagem não encontrada em: {caminho_imagem}")
    
    # Criar pasta para resultados
    pasta_resultados = criar_pasta_resultados(caminho_imagem)
    
    # Pré-processamento
    pre_processador = PipelinePreProcessamento()
    img_processada = pre_processador.executar_pipeline(img_original, False, pasta_resultados)
    
    # Segmentação
    segmentador = PipelineSegmentacao(
        min_area=50,
        max_area=100
    )
    # A pipeline de segmentação agora retorna a imagem binária e os contornos
    _, contornos = segmentador.executar_pipeline(img_processada, False, pasta_resultados)
    
    # Contagem
    contador = OperacoesContagem()
    total_contado = contador.contar_objetos(contornos)
    print(f"Total de microorganismos contados: {total_contado}")
    
    # Desenhar resultados na imagem original
    img_final_com_contagem = contador.desenhar_resultados(img_original, contornos, total_contado)
    
    # Salvar a imagem final
    salvar_imagem(pasta_resultados, "08_resultado_final_com_contagem", img_final_com_contagem)

    # Exibir resultado final
    if mostrar_resultados:
        cv2.imshow("Resultado Final com Contagem", img_final_com_contagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img_final_com_contagem, total_contado

if __name__ == "__main__":
    nome_arquivo = "exemplo.jpg"
    caminho_imagem = obter_caminho_imagem(nome_arquivo)
    executar_pipeline(caminho_imagem)