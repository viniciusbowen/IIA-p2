"""
Grad-CAM Module - Projeto 2 IIA (UnB 2025/2)
Visualização de regiões importantes usando Gradient-weighted Class Activation Maps

Aplicado ao Discriminador do pix2pix para interpretar decisões de anomalia.

Referência: Selvaraju et al. (2017)

Autor: Sistema Automático
Data: Dezembro 2025
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, Optional


class GradCAMVisualizer:
    """Gerador de Grad-CAM para visualização de features importantes."""
    
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        """
        Inicializa o Grad-CAM.
        
        Args:
            model: Modelo TensorFlow (normalmente o discriminador)
            layer_name: Nome da camada para extrair features
                       Se None, usa a última camada conv antes do output
        """
        self.model = model
        self.layer_name = layer_name
        
        # Se não especificado, encontra última camada convolucional
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    self.layer_name = layer.name
                    break
    
    def compute_gradcam(self, images: np.ndarray) -> np.ndarray:
        """
        Calcula Grad-CAM para um batch de imagens.
        
        Processo:
        1. Forward pass até a layer especificada
        2. Calcular gradientes em relação aos feature maps
        3. Ponderar feature maps pelos gradientes
        4. Fazer upsample para tamanho da imagem original
        
        Args:
            images: [N, H, W, 3] imagens normalizadas [0,1]
            
        Returns:
            [N, H, W] mapas de ativação normalizados [0,1]
        """
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Obter a camada de interesse
        target_layer = self.model.get_layer(self.layer_name)
        
        # Para modelos subclassed, precisamos criar um extractor diferente
        # Usar GradientTape para calcular gradientes diretamente
        heatmaps = []
        
        for i in range(images_tensor.shape[0]):
            img = images_tensor[i:i+1]
            
            with tf.GradientTape() as tape:
                # Criar um modelo intermediário para capturar features
                # Precisamos registrar a imagem para o tape
                tape.watch(img)
                
                # Forward pass - capturar ativações da camada intermediária
                # Usar um extractor temporário
                layer_output = None
                
                # Hook para capturar output da camada
                @tf.custom_gradient
                def capture_and_forward(x):
                    nonlocal layer_output
                    # Fazer forward pass manual através das camadas
                    current = x
                    for layer in self.model.layers:
                        current = layer(current)
                        if layer.name == self.layer_name:
                            layer_output = current
                    
                    def grad(dy):
                        return dy
                    return current, grad
                
                # Alternativa mais simples: usar tf.GradientTape nested
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(img)
                    
                    # Forward pass capturando ativações
                    activations = []
                    x = img
                    for layer in self.model.layers:
                        x = layer(x)
                        if layer.name == self.layer_name:
                            activations.append(x)
                    
                    output = x
                    features = activations[0] if activations else x
                
                # Calcular gradientes
                grads = inner_tape.gradient(output, features)
            
            if grads is None or features is None:
                # Fallback: retornar heatmap uniforme
                heatmaps.append(np.ones((images.shape[1], images.shape[2])) * 0.5)
                continue
            
            # Ponderar features pelos gradientes (Global Average Pooling dos gradientes)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Aplicar pesos aos feature maps
            features_squeezed = features[0]  # [H, W, filters]
            weighted_features = features_squeezed * pooled_grads
            
            # Somar canais para obter heatmap
            heatmap = tf.reduce_sum(weighted_features, axis=-1)
            
            # ReLU - manter apenas ativações positivas
            heatmap = tf.nn.relu(heatmap)
            
            # Normalizar para [0, 1]
            heatmap = heatmap.numpy()
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Redimensionar para tamanho da imagem original
            heatmap_resized = cv2.resize(heatmap, (images.shape[2], images.shape[1]))
            heatmaps.append(heatmap_resized)
        
        return np.array(heatmaps)
    
    def overlay_gradcam(self, image: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4, cmap: str = 'jet') -> np.ndarray:
        """
        Sobrepõe Grad-CAM na imagem original.
        
        Args:
            image: Imagem original [H, W, 3] em [0,1]
            heatmap: Grad-CAM [H, W] em [0,1]
            alpha: Transparência do heatmap (0-1)
            cmap: Mapa de cores
            
        Returns:
            Imagem com overlay [H, W, 3] em [0,1]
        """
        # Upscale heatmap se necessário
        image_h, image_w = image.shape[:2]
        heatmap_h, heatmap_w = heatmap.shape[:2]
        
        if (image_h, image_w) != (heatmap_h, heatmap_w):
            heatmap = cv2.resize(heatmap, (image_w, image_h))
        
        # Normalizar
        image = np.clip(image, 0, 1)
        heatmap = np.clip(heatmap, 0, 1)
        
        # Converter para uint8
        image_uint8 = (image * 255).astype(np.uint8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Aplicar colormap
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
        }
        
        cmap_code = colormap_dict.get(cmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cmap_code)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blending
        overlaid = cv2.addWeighted(image_uint8, 1 - alpha, 
                                   heatmap_colored_rgb, alpha, 0)
        
        return overlaid.astype(np.float32) / 255.0
    
    def batch_gradcam(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula Grad-CAM para batch de imagens com overlay.
        
        Args:
            images: [N, H, W, 3]
            
        Returns:
            Tupla (heatmaps, overlaid_images)
        """
        heatmaps = []
        overlays = []
        
        for i in range(len(images)):
            heatmap = self.compute_gradcam(images[i:i+1])
            overlay = self.overlay_gradcam(images[i], heatmap[0])
            
            heatmaps.append(heatmap[0])
            overlays.append(overlay)
        
        return np.array(heatmaps), np.array(overlays)


class InterpretabilityVisualizer:
    """Visualizador de interpretabilidade: combina reconstrução, anomalia e Grad-CAM."""
    
    @staticmethod
    def create_comparison_grid(original: np.ndarray, 
                              reconstructed: np.ndarray,
                              anomaly_map: np.ndarray,
                              gradcam_overlay: np.ndarray,
                              figsize: Tuple[int, int] = (16, 4)) -> np.ndarray:
        """
        Cria grid comparativo: Original | Reconstruído | Anomalia | Grad-CAM.
        
        Args:
            original: [H, W, 3]
            reconstructed: [H, W, 3]
            anomaly_map: [H, W]
            gradcam_overlay: [H, W, 3]
            figsize: (width, height) em polegadas
            
        Returns:
            Grid em escala [0,255]
        """
        # Garantir tamanho igual
        h, w = original.shape[:2]
        
        # Redimensionar se necessário
        reconstructed = cv2.resize(reconstructed, (w, h)) if reconstructed.shape != original.shape else reconstructed
        anomaly_map = cv2.resize(anomaly_map, (w, h)) if anomaly_map.shape != original.shape[:2] else anomaly_map
        gradcam_overlay = cv2.resize(gradcam_overlay, (w, h)) if gradcam_overlay.shape != original.shape else gradcam_overlay
        
        # Converter anomaly map para RGB (cinza)
        anomaly_rgb = np.stack([anomaly_map] * 3, axis=-1)
        
        # Concatenar horizontalmente
        grid = np.hstack([original, reconstructed, anomaly_rgb, gradcam_overlay])
        
        # Converter para uint8
        grid_uint8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
        
        return grid_uint8


if __name__ == "__main__":
    print("🔧 Testando módulo Grad-CAM...")
    
    # Dummy discriminator
    dummy_input = tf.keras.layers.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(dummy_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    output = tf.keras.layers.Conv2D(1, 4, padding='same')(x)
    
    dummy_discriminator = tf.keras.Model(inputs=dummy_input, outputs=output)
    
    visualizer = GradCAMVisualizer(dummy_discriminator)
    
    # Teste com dummy image
    dummy_image = np.random.randn(1, 256, 256, 3)
    dummy_image = np.clip(dummy_image * 0.1 + 0.5, 0, 1)
    
    heatmap = visualizer.compute_gradcam(dummy_image)
    print(f"✅ Grad-CAM shape: {heatmap.shape}")
    print(f"✅ Grad-CAM range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
