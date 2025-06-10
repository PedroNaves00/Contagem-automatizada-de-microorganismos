import cv2
import numpy as np
from typing import Optional, Tuple

class PreProcessador:
    
    def __init__(self, 
                 kernel_size: Tuple[int, int] = (5, 5),
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 threshold_value: int = 127):
        """
        Inicializa o pré-processador com parâmetros configuráveis.
        
        Args:
            kernel_size: Tamanho do kernel para o desfoque gaussiano
            clahe_clip_limit: Limite de clip para o CLAHE
            clahe_grid_size: Tamanho da grade para o CLAHE
            threshold_value: Valor do limiar para binarização
        """
        self.kernel_size = kernel_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.threshold_value = threshold_value

    def _validar_imagem(self, img: np.ndarray) -> None:
        """
        Valida se a imagem de entrada é válida.
        """
        if img is None:
            raise ValueError("Imagem de entrada não pode ser None")
        if len(img.shape) not in [2, 3]:
            raise ValueError("Imagem deve ser 2D (tons de cinza) ou 3D (colorida)")

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