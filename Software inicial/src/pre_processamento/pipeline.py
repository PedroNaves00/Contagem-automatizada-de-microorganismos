import cv2
import numpy as np
from .operacoes import OperacoesPreProcessamento
from ..utils import salvar_imagem

class PipelinePreProcessamento(OperacoesPreProcessamento):
    """
    Classe que implementa o pipeline completo de pré-processamento.
    """

    def executar_pipeline(self, img: np.ndarray, mostrar_resultados: bool = False, pasta_resultados: str = None) -> np.ndarray:
        """
        Executa o pipeline completo de pré-processamento.
        Args:
            img: Imagem de entrada
            mostrar_resultados: Se True, exibe as imagens intermediárias
            pasta_resultados: Caminho da pasta para salvar os resultados
        Returns:
            Imagem final processada
        """
        self._validar_imagem(img)
        
        # Pipeline de processamento
        img_cinza = self.para_cinza(img)
        if mostrar_resultados:
            self.mostrar_imagem("Imagem em Tons de Cinza", img_cinza)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "01_cinza", img_cinza)
            
        img_desfocada = self.desfocar(img_cinza)
        if mostrar_resultados:
            self.mostrar_imagem("Imagem Desfocada", img_desfocada)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "02_desfocada", img_desfocada)
            
        img_realce = self.realce_contraste(img_desfocada)
        if mostrar_resultados:
            self.mostrar_imagem("Imagem com CLAHE", img_realce)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "03_realce", img_realce)
            
        img_binaria = self.limiarizar(img_realce)
        if mostrar_resultados:
            self.mostrar_imagem("Imagem Limiarizada", img_binaria)
        if pasta_resultados:
            salvar_imagem(pasta_resultados, "04_binaria", img_binaria)
            
        return img_binaria 