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
            img: Imagem binária de entrada.
            mostrar_resultados: Se True, exibe as imagens intermediárias.
            pasta_resultados: Caminho da pasta para salvar os resultados.
            
        Returns:
            Tupla contendo a imagem binária e a lista de contornos filtrados.
        """
        self._validar_imagem(img)
        
        # Encontrar contornos
        contornos = self.encontrar_contornos(img)
        
        # Filtrar contornos
        contornos_filtrados = self.filtrar_contornos(contornos)
        
        if mostrar_resultados:
            img_com_contornos = self.desenhar_contornos(img.copy(), contornos_filtrados)
            self.mostrar_imagem("Contornos Filtrados", img_com_contornos)
        
        if pasta_resultados:
            img_para_salvar = self.desenhar_contornos(img.copy(), contornos_filtrados)
            salvar_imagem(pasta_resultados, "06_contornos_filtrados", img_para_salvar)
            
        return img, contornos_filtrados