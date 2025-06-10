import cv2
import numpy as np
from typing import List, Tuple

class Segmentador:
    """
    Classe base para segmentação de microorganismos.
    """

    def __init__(self, min_area: int = 100, max_area: int = 1000):
        """
        Inicializa o segmentador.
        Args:
            min_area: Área mínima para considerar um objeto
            max_area: Área máxima para considerar um objeto
        """
        self.min_area = min_area
        self.max_area = max_area

    def _validar_imagem(self, img: np.ndarray) -> None:
        """
        Valida se a imagem de entrada é válida.
        Args:
            img: Imagem a ser validada
        Raises:
            ValueError: Se a imagem for inválida
        """
        if img is None:
            raise ValueError("Imagem de entrada não pode ser None")
        if len(img.shape) != 2:
            raise ValueError("Imagem deve ser binária (2D)")

    @staticmethod
    def mostrar_imagem(titulo: str, img: np.ndarray) -> None:
        """
        Exibe uma imagem em uma janela.
        Args:
            titulo: Título da janela
            img: Imagem a ser exibida
        """
        cv2.imshow(titulo, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 