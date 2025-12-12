"""
Anomaly Detection Module - Projeto 2 IIA (UnB 2025/2)
Cálculo do índice de anomalia e geração de mapas

Implementação exata conforme guia:
    A(x,y) = ||I(x,y) - R(x,y)||²

Onde:
    I(x,y) = imagem original
    R(x,y) = imagem reconstruída
    A(x,y) = índice de anomalia por pixel

Autor: Sistema Automático
Data: Dezembro 2025
"""

import numpy as np
import tensorflow as tf
from typing import Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2


class AnomalyDetector:
    """Detector de anomalias baseado em discrepâncias pix2pix."""
    
    def __init__(self, threshold_method: str = 'otsu'):
        """
        Inicializa o detector.
        
        Args:
            threshold_method: 'otsu' ou 'percentile' para binarização
        """
        self.threshold_method = threshold_method
    
    def compute_anomaly_map(self, original: np.ndarray, reconstructed: np.ndarray,
                           return_normalized: bool = True) -> Tuple[np.ndarray, float]:
        """
        Calcula o mapa de anomalia pixel-a-pixel.
        
        Fórmula exata do guia:
            A(x,y) = ||I(x,y) - R(x,y)||²
        
        Args:
            original: Imagem original [H, W, 3] em [0,1]
            reconstructed: Imagem reconstruída [H, W, 3] em [0,1]
            return_normalized: Se normalizar para [0,1]
            
        Returns:
            Tupla (mapa_anomalia, valor_global)
        """
        # Garantir formato correto
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # Diferença quadrada (L2): ||I - R||²
        diff = original.astype(np.float32) - reconstructed.astype(np.float32)
        anomaly_map = np.sum(diff ** 2, axis=-1)  # Somar canais RGB
        
        # Valor global (média)
        global_anomaly = float(np.mean(anomaly_map))
        
        # Normalizar para [0,1] se desejado
        if return_normalized:
            max_val = np.max(anomaly_map)
            if max_val > 0:
                anomaly_map_norm = anomaly_map / max_val
            else:
                anomaly_map_norm = anomaly_map
            return anomaly_map_norm, global_anomaly
        
        return anomaly_map, global_anomaly
    
    def batch_anomaly_maps(self, originals: np.ndarray, 
                          reconstructed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula anomalias para um batch de imagens.
        
        Args:
            originals: [N, H, W, 3]
            reconstructed: [N, H, W, 3]
            
        Returns:
            Tupla (mapas_anomalia, valores_globais)
        """
        maps = []
        globals = []
        
        for i in range(len(originals)):
            amap, aglobal = self.compute_anomaly_map(originals[i], reconstructed[i])
            maps.append(amap)
            globals.append(aglobal)
        
        return np.array(maps), np.array(globals)
    
    def compute_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calcula PSNR (Peak Signal-to-Noise Ratio).
        
        Métrica de qualidade: quanto maior, melhor.
        Típico: 20-50 dB (acima de 30 é muito bom)
        
        Args:
            original: Imagem original [0,1]
            reconstructed: Imagem reconstruída [0,1]
            
        Returns:
            PSNR em dB
        """
        # Converter para [0,1] se necessário
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # Usar valores [0,255] para cálculo padrão
        original_8bit = (original * 255).astype(np.uint8)
        reconstructed_8bit = (reconstructed * 255).astype(np.uint8)
        
        psnr_val = psnr(original_8bit, reconstructed_8bit, data_range=255)
        return float(psnr_val)
    
    def compute_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calcula SSIM (Structural Similarity Index).
        
        Métrica perceptual: varia de -1 a 1 (1 = idêntico).
        Valores típicos: 0.7-1.0 para boa qualidade.
        
        Args:
            original: Imagem original [0,1]
            reconstructed: Imagem reconstruída [0,1]
            
        Returns:
            SSIM em [-1, 1]
        """
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # SSIM requer valores [0,255]
        original_8bit = (original * 255).astype(np.uint8)
        reconstructed_8bit = (reconstructed * 255).astype(np.uint8)
        
        if len(original_8bit.shape) == 3:  # Color image
            ssim_val = ssim(original_8bit, reconstructed_8bit, 
                           channel_axis=2, data_range=255)
        else:  # Grayscale
            ssim_val = ssim(original_8bit, reconstructed_8bit, data_range=255)
        
        return float(ssim_val)
    
    def batch_metrics(self, originals: np.ndarray, 
                     reconstructed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula PSNR e SSIM para batch de imagens.
        
        Args:
            originals: [N, H, W, 3]
            reconstructed: [N, H, W, 3]
            
        Returns:
            Tupla (psnr_array, ssim_array)
        """
        psnr_values = []
        ssim_values = []
        
        for i in range(len(originals)):
            psnr_val = self.compute_psnr(originals[i], reconstructed[i])
            ssim_val = self.compute_ssim(originals[i], reconstructed[i])
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
        
        return np.array(psnr_values), np.array(ssim_values)
    
    def threshold_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Binariza o mapa de anomalia usando threshold.
        
        Args:
            anomaly_map: Mapa normalizado [0,1]
            
        Returns:
            Mapa binarizado [0,1]
        """
        if self.threshold_method == 'otsu':
            # Otsu: encontra threshold ótimo
            anomaly_8bit = (anomaly_map * 255).astype(np.uint8)
            _, binary = cv2.threshold(anomaly_8bit, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary.astype(np.float32) / 255.0
        
        elif self.threshold_method == 'percentile':
            # Percentil (padrão: 75º percentil)
            threshold = np.percentile(anomaly_map, 75)
            return (anomaly_map > threshold).astype(np.float32)
        
        else:
            raise ValueError(f"Método desconhecido: {self.threshold_method}")
    
    def colormap_anomaly(self, anomaly_map: np.ndarray, cmap: str = 'jet') -> np.ndarray:
        """
        Converte mapa de anomalia em imagem colorida para visualização.
        
        Args:
            anomaly_map: Mapa em escala de cinza [0,1]
            cmap: Mapa de cores ('jet', 'hot', 'viridis', etc)
            
        Returns:
            Imagem colorida [H, W, 3] em [0,1]
        """
        # Normalizar
        anomaly_normalized = np.clip(anomaly_map, 0, 1)
        
        # Converter para uint8
        anomaly_8bit = (anomaly_normalized * 255).astype(np.uint8)
        
        # Aplicar colormap
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA
        }
        
        if cmap not in colormap_dict:
            cmap = 'jet'
        
        colored = cv2.applyColorMap(anomaly_8bit, colormap_dict[cmap])
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored_rgb.astype(np.float32) / 255.0


if __name__ == "__main__":
    print("🔧 Testando módulo de detecção de anomalias...")
    
    detector = AnomalyDetector()
    
    # Dummy images
    original = np.random.randn(256, 256, 3) * 0.1 + 0.5
    original = np.clip(original, 0, 1)
    
    reconstructed = original + np.random.randn(256, 256, 3) * 0.05
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Testes
    amap, aglobal = detector.compute_anomaly_map(original, reconstructed)
    psnr_val = detector.compute_psnr(original, reconstructed)
    ssim_val = detector.compute_ssim(original, reconstructed)
    
    print(f"✅ Mapa de anomalia shape: {amap.shape}")
    print(f"✅ Valor global de anomalia: {aglobal:.6f}")
    print(f"✅ PSNR: {psnr_val:.2f} dB")
    print(f"✅ SSIM: {ssim_val:.4f}")
