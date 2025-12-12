"""
Utils Module - Projeto 2 IIA (UnB 2025/2)
Funções auxiliares para visualização, salvar/carregar modelos, etc.

Autor: Sistema Automático
Data: Dezembro 2025
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle


class ModelManager:
    """Gerenciador de salvar/carregar modelos."""
    
    @staticmethod
    def save_model(model: tf.keras.Model, save_path: str):
        """Salva modelo em formato SavedModel."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        model.save(save_path)
        print(f"✅ Modelo salvo em: {save_path}")
    
    @staticmethod
    def load_model(save_path: str) -> tf.keras.Model:
        """Carrega modelo em formato SavedModel."""
        model = tf.keras.models.load_model(save_path)
        print(f"✅ Modelo carregado de: {save_path}")
        return model
    
    @staticmethod
    def save_weights(model: tf.keras.Model, save_path: str):
        """Salva pesos do modelo."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        model.save_weights(save_path)
        print(f"✅ Pesos salvos em: {save_path}")
    
    @staticmethod
    def load_weights(model: tf.keras.Model, save_path: str):
        """Carrega pesos do modelo."""
        model.load_weights(save_path)
        print(f"✅ Pesos carregados de: {save_path}")
        return model


class ImageVisualizer:
    """Visualizador de imagens e resultados."""
    
    @staticmethod
    def plot_images(images: List[np.ndarray], titles: List[str] = None,
                   figsize: Tuple[int, int] = (15, 5), save_path: str = None):
        """
        Plota múltiplas imagens em grid.
        
        Args:
            images: Lista de imagens [H, W, 3] ou [H, W]
            titles: Lista de títulos
            figsize: Tamanho da figura
            save_path: Se fornecido, salva a figura
        """
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        
        if n == 1:
            axes = [axes]
        
        for i, img in enumerate(images):
            ax = axes[i]
            
            # Garantir que está em [0,1] ou [0,255]
            if img.max() > 1:
                img = img / 255.0
            
            # Plotar
            if len(img.shape) == 3 and img.shape[2] == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figura salva em: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_distributions(values_list: List[np.ndarray], labels: List[str],
                          title: str = "Distribuição de Valores",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: str = None):
        """
        Plota histogramas de distribuições.
        
        Args:
            values_list: Lista de arrays de valores
            labels: Rótulos para cada array
            title: Título do gráfico
            figsize: Tamanho da figura
            save_path: Se fornecido, salva a figura
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for values, label in zip(values_list, labels):
            ax.hist(values, bins=30, alpha=0.6, label=label)
        
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figura salva em: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                      title: str = "Curva ROC",
                      figsize: Tuple[int, int] = (8, 8),
                      save_path: str = None):
        """
        Plota curva ROC.
        
        Args:
            fpr: Taxa de falsos positivos
            tpr: Taxa de verdadeiros positivos
            auc: Área sob a curva
            title: Título do gráfico
            figsize: Tamanho da figura
            save_path: Se fornecido, salva a figura
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2, color='blue')
        ax.plot([0, 1], [0, 1], 'r--', label='Classificador Aleatório', linewidth=2)
        
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figura salva em: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history: dict, save_path: str = None):
        """
        Plota histórico de treinamento (perdas).
        
        Args:
            history: Dicionário com 'g_loss', 'd_loss', 'l1_loss'
            save_path: Se fornecido, salva a figura
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        epochs = range(1, len(history['g_loss']) + 1)
        
        ax.plot(epochs, history['g_loss'], 'o-', label='Generator Loss', linewidth=2)
        ax.plot(epochs, history['d_loss'], 's-', label='Discriminator Loss', linewidth=2)
        if 'l1_loss' in history:
            ax.plot(epochs, history['l1_loss'], '^-', label='L1 Loss', linewidth=2)
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.set_title('Histórico de Treinamento')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figura salva em: {save_path}")
        
        plt.show()


class DataProcessor:
    """Processador de dados e arrays."""
    
    @staticmethod
    def normalize_batch(images: np.ndarray, target_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """
        Normaliza batch de imagens para um intervalo.
        
        Args:
            images: [N, H, W, 3] em [0,1]
            target_range: (min, max) do intervalo alvo
            
        Returns:
            Imagens normalizadas no intervalo alvo
        """
        images = np.clip(images, 0, 1)
        min_val, max_val = target_range
        return images * (max_val - min_val) + min_val
    
    @staticmethod
    def denormalize_batch(images: np.ndarray, source_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """
        Denormaliza batch de imagens de volta para [0,1].
        
        Args:
            images: [N, H, W, 3] no intervalo source_range
            source_range: (min, max) do intervalo original
            
        Returns:
            Imagens em [0,1]
        """
        min_val, max_val = source_range
        return (images - min_val) / (max_val - min_val)
    
    @staticmethod
    def save_array(data: np.ndarray, save_path: str):
        """Salva array como arquivo .npy."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.save(save_path, data)
        print(f"✅ Array salvo em: {save_path}")
    
    @staticmethod
    def load_array(save_path: str) -> np.ndarray:
        """Carrega array de arquivo .npy."""
        data = np.load(save_path)
        print(f"✅ Array carregado de: {save_path}")
        return data
    
    @staticmethod
    def save_dict(data: dict, save_path: str):
        """Salva dicionário como pickle."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Dicionário salvo em: {save_path}")
    
    @staticmethod
    def load_dict(save_path: str) -> dict:
        """Carrega dicionário de arquivo pickle."""
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Dicionário carregado de: {save_path}")
        return data


if __name__ == "__main__":
    print("✅ Utils module loaded")
