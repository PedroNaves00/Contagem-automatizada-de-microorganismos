import cv2
import numpy as np
from typing import List, Tuple
from .base import Segmentador

class OperacoesSegmentacao(Segmentador):
    """
    Classe que implementa as operações específicas de segmentação.
    """

    def encontrar_contornos(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Encontra os contornos dos objetos na imagem binária.
        Args:
            img: Imagem binária
        Returns:
            Lista de contornos encontrados
        """
        self._validar_imagem(img)
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contornos

    def filtrar_contornos(self, contornos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filtra os contornos baseado na área.
        Args:
            contornos: Lista de contornos
        Returns:
            Lista de contornos filtrados
        """
        contornos_filtrados = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if self.min_area <= area <= self.max_area:
                contornos_filtrados.append(contorno)
        return contornos_filtrados

    def desenhar_contornos(self, img: np.ndarray, contornos: List[np.ndarray]) -> np.ndarray:
        """
        Desenha os contornos na imagem.
        Args:
            img: Imagem original
            contornos: Lista de contornos
        Returns:
            Imagem com os contornos desenhados
        """
        img_contornos = img.copy()
        cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
        return img_contornos 