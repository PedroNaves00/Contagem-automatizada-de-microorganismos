import cv2
import numpy as np
from .base import PreProcessador

class OperacoesPreProcessamento(PreProcessador):
    """
    Classe que implementa as operações específicas de pré-processamento.
    """

    def para_cinza(self, img: np.ndarray) -> np.ndarray:
        """
        Converte a imagem para tons de cinza.
        Args:
            img: Imagem colorida em formato RGB
        Returns:
            Imagem em tons de cinza
        """
        self._validar_imagem(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def desfocar(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica desfoque gaussiano na imagem.
        Args:
            img: Imagem em tons de cinza
        Returns:
            Imagem desfocada
        """
        self._validar_imagem(img)
        return cv2.GaussianBlur(img, self.kernel_size, 0)

    def realce_contraste(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica realce de contraste usando CLAHE.
        Args:
            img: Imagem em tons de cinza
        Returns:
            Imagem com contraste realçado
        """
        self._validar_imagem(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        return clahe.apply(img)

    def limiarizar(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica limiarização na imagem.
        Args:
            img: Imagem em tons de cinza
        Returns:
            Imagem binarizada
        """
        self._validar_imagem(img)
        _, img_bin = cv2.threshold(img, self.threshold_value, 255, cv2.THRESH_BINARY)
        return img_bin 