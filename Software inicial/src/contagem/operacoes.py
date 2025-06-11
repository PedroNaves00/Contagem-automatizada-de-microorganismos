import cv2
import numpy as np
from typing import List

class OperacoesContagem:
    """
    Classe para realizar a contagem de objetos e exibir os resultados.
    """

    def contar_objetos(self, contornos: List[np.ndarray]) -> int:
        """
        Retorna o número de contornos detectados.
        
        Args:
            contornos: Lista de contornos filtrados.
            
        Returns:
            Número total de objetos contados.
        """
        return len(contornos)

    def desenhar_resultados(self, img: np.ndarray, contornos: List[np.ndarray], total_contado: int) -> np.ndarray:
        """
        Desenha os contornos, numeração e o total contado na imagem.
        
        Args:
            img: Imagem original (ou a cópia para desenho).
            contornos: Lista de contornos a serem desenhados.
            total_contado: O número total de objetos.
            
        Returns:
            Imagem com os resultados desenhados.
        """
        img_resultado = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

        # Desenha e numera cada contorno
        for i, contorno in enumerate(contornos):
            # Desenha o contorno
            cv2.drawContours(img_resultado, [contorno], -1, (0, 255, 0), 2)
            
            # Calcula o centroide para posicionar o número
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Se o momento for 0, usa o ponto superior do contorno
                cX, cY = contorno[0][0]

            # Desenha o número do objeto
            cv2.putText(img_resultado, f"#{i+1}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 0, 0), 2)

        # Desenha o total contado no canto superior da imagem
        texto_total = f"Total de Microorganismos: {total_contado}"
        cv2.putText(img_resultado, texto_total, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return img_resultado