import cv2
import numpy as np
from typing import List, Tuple
from .operacoes import OperacoesSegmentacao
from ..utils import salvar_imagem

class PipelineSegmentacao(OperacoesSegmentacao):
    """
    Classe que implementa o pipeline completo de segmentação.
    """

    def executar_pipeline(self, img: np.ndarray, mostrar_resultados: bool = False, pasta_resultados: str = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executa o pipeline completo de segmentação.
        
        Args:
            img: Imagem binária de entrada
            mostrar_resultados: Se True, exibe as imagens intermediárias
            pasta_resultados: Caminho da pasta para salvar os resultados
            
        Returns:
            Tupla contendo a imagem com contornos desenhados e a lista de contornos
        """
        self._validar_imagem(img)
        
        # Encontrar contornos
        contornos = self.encontrar_contornos(img)
        img_contornos = self.desenhar_contornos(img, contornos)
        if mostrar_resultados:
            self.mostrar_imagem("Contornos Encontrados", img_contornos)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "05_contornos_iniciais", img_contornos)
        
        # Filtrar contornos
        contornos_filtrados = self.filtrar_contornos(contornos)
        img_filtrada = self.desenhar_contornos(img, contornos_filtrados)
        if mostrar_resultados:
            self.mostrar_imagem("Contornos Filtrados", img_filtrada)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "06_contornos_filtrados", img_filtrada)
        
        # Desenhar contornos finais
        img_final = self.desenhar_contornos(img, contornos_filtrados)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "07_resultado_final", img_final)
        
        return img_final, contornos_filtrados 