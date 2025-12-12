"""
Data Loader Module - Projeto 2 IIA (UnB 2025/2)
Carrega e preprocessa imagens de folhas (saudáveis e doentes)

Autor: Sistema Automático
Data: Dezembro 2025
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List


class DataLoader:
    """Carregador de dataset de folhas com suporte a normalização e augmentation."""
    
    def __init__(self, image_size: int = 256):
        """
        Inicializa o Data Loader.
        
        Args:
            image_size: Tamanho das imagens (256x256 padrão)
        """
        self.image_size = image_size
        
    def load_images_from_folder(self, folder_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Carrega todas as imagens de uma pasta.
        
        Args:
            folder_path: Caminho da pasta com imagens
            
        Returns:
            Tupla (array_imagens, lista_nomes)
        """
        images = []
        names = []
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Pasta não encontrada: {folder_path}")
            return np.array([]), []
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                filepath = os.path.join(folder_path, filename)
                
                # Carregar imagem em BGR
                img = cv2.imread(filepath)
                
                if img is not None:
                    # Converter BGR → RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Redimensionar
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    # Normalizar para [-1, 1] (padrão pix2pix com tanh)
                    img = img.astype(np.float32)
                    img = (img / 127.5) - 1.0
                    
                    images.append(img)
                    names.append(filename)
                else:
                    print(f"❌ Erro ao carregar: {filename}")
        
        return np.array(images), names
    
    def load_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     List[str], List[str], List[str]]:
        """
        Carrega o dataset completo (treino, teste saudável, teste doente).
        
        Estrutura esperada:
        data/
          ├── train_healthy/
          ├── test_healthy/
          └── test_diseased/
        
        Args:
            data_dir: Diretório raiz dos dados
            
        Returns:
            Tupla (X_train, X_test_healthy, X_test_diseased, 
                   names_train, names_test_h, names_test_d)
        """
        train_path = os.path.join(data_dir, "train_healthy")
        test_healthy_path = os.path.join(data_dir, "test_healthy")
        test_diseased_path = os.path.join(data_dir, "test_diseased")
        
        print("📂 Carregando dataset...")
        
        # Carregar dados
        X_train, names_train = self.load_images_from_folder(train_path)
        X_test_healthy, names_test_h = self.load_images_from_folder(test_healthy_path)
        X_test_diseased, names_test_d = self.load_images_from_folder(test_diseased_path)
        
        print(f"✅ Treino (saudáveis): {len(X_train)} imagens")
        print(f"✅ Teste (saudáveis): {len(X_test_healthy)} imagens")
        print(f"✅ Teste (doentes): {len(X_test_diseased)} imagens")
        
        return X_train, X_test_healthy, X_test_diseased, names_train, names_test_h, names_test_d
    
    def create_tf_dataset(self, X: np.ndarray, batch_size: int = 16, 
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Cria um Dataset TensorFlow otimizado.
        
        Args:
            X: Array de imagens
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar
            
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(X)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


if __name__ == "__main__":
    # Teste rápido
    loader = DataLoader(image_size=256)
    
    # Ajuste o caminho conforme sua estrutura
    data_path = "../data"
    
    if os.path.exists(data_path):
        X_train, X_test_h, X_test_d, _, _, _ = loader.load_dataset(data_path)
        print(f"\n📊 Shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_test_healthy: {X_test_h.shape}")
        print(f"   X_test_diseased: {X_test_d.shape}")
    else:
        print(f"⚠️  Diretório não encontrado: {data_path}")
