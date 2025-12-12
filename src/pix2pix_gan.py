"""
pix2pix GAN Module - Projeto 2 IIA (UnB 2025/2)
Implementação do pix2pix para detecção de anomalias em folhas

Referência: Isola et al. (2017) - Image-to-Image Translation with Conditional GANs
Baseado em: Katafuchi & Tokunaga (2020)

Autor: Sistema Automático
Data: Dezembro 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Callable


class Pix2PixGenerator(keras.Model):
    """
    Generator U-Net para pix2pix.
    
    Arquitetura:
    - Encoder: downsample com Conv2D
    - Bottleneck: camadas de convolução
    - Decoder: upsample com Conv2DTranspose
    - Skip connections entre encoder/decoder
    """
    
    def __init__(self, image_size: int = 256):
        super().__init__()
        self.image_size = image_size
        
        # ===== ENCODER (Downsampling) =====
        self.enc1 = self._conv_block(64, kernel_size=4, strides=2, norm=False)   # 256 → 128
        self.enc2 = self._conv_block(128, kernel_size=4, strides=2, norm=True)   # 128 → 64
        self.enc3 = self._conv_block(256, kernel_size=4, strides=2, norm=True)   # 64 → 32
        self.enc4 = self._conv_block(512, kernel_size=4, strides=2, norm=True)   # 32 → 16
        self.enc5 = self._conv_block(512, kernel_size=4, strides=2, norm=True)   # 16 → 8
        self.enc6 = self._conv_block(512, kernel_size=4, strides=2, norm=True)   # 8 → 4
        self.enc7 = self._conv_block(512, kernel_size=4, strides=2, norm=True)   # 4 → 2
        
        # ===== BOTTLENECK =====
        self.bottleneck = self._conv_block(512, kernel_size=4, strides=2, norm=True)  # 2 → 1
        
        # ===== DECODER (Upsampling com skip connections) =====
        self.dec7 = self._deconv_block(512, kernel_size=4, strides=2, dropout=0.5)
        self.dec6 = self._deconv_block(512, kernel_size=4, strides=2, dropout=0.5)
        self.dec5 = self._deconv_block(512, kernel_size=4, strides=2, dropout=0.5)
        self.dec4 = self._deconv_block(512, kernel_size=4, strides=2)
        self.dec3 = self._deconv_block(256, kernel_size=4, strides=2)
        self.dec2 = self._deconv_block(128, kernel_size=4, strides=2)
        self.dec1 = self._deconv_block(64, kernel_size=4, strides=2)
        
        # ===== OUTPUT =====
        self.final = layers.Conv2DTranspose(3, kernel_size=4, strides=2, 
                                           padding='same', activation='tanh')
    
    def _conv_block(self, filters: int, kernel_size: int = 4, strides: int = 2, 
                   norm: bool = True) -> keras.Sequential:
        """Bloco convolucional: Conv2D + BatchNorm + LeakyReLU"""
        block = keras.Sequential([
            layers.Conv2D(filters, kernel_size, strides, padding='same', 
                         use_bias=not norm)
        ])
        if norm:
            block.add(layers.BatchNormalization())
        block.add(layers.LeakyReLU(0.2))
        return block
    
    def _deconv_block(self, filters: int, kernel_size: int = 4, strides: int = 2,
                     dropout: float = 0.0) -> keras.Sequential:
        """Bloco deconvolucional: Conv2DTranspose + BatchNorm + (Dropout) + ReLU"""
        block = keras.Sequential([
            layers.Conv2DTranspose(filters, kernel_size, strides, padding='same'),
            layers.BatchNormalization(),
        ])
        if dropout > 0:
            block.add(layers.Dropout(dropout))
        block.add(layers.ReLU())
        return block
    
    def call(self, x, training=False):
        """Forward pass com skip connections"""
        # Encoder
        e1 = self.enc1(x, training=training)
        e2 = self.enc2(e1, training=training)
        e3 = self.enc3(e2, training=training)
        e4 = self.enc4(e3, training=training)
        e5 = self.enc5(e4, training=training)
        e6 = self.enc6(e5, training=training)
        e7 = self.enc7(e6, training=training)
        
        # Bottleneck
        b = self.bottleneck(e7, training=training)
        
        # Decoder com skip connections
        d7 = self.dec7(b, training=training)
        d7 = layers.concatenate([d7, e7])
        
        d6 = self.dec6(d7, training=training)
        d6 = layers.concatenate([d6, e6])
        
        d5 = self.dec5(d6, training=training)
        d5 = layers.concatenate([d5, e5])
        
        d4 = self.dec4(d5, training=training)
        d4 = layers.concatenate([d4, e4])
        
        d3 = self.dec3(d4, training=training)
        d3 = layers.concatenate([d3, e3])
        
        d2 = self.dec2(d3, training=training)
        d2 = layers.concatenate([d2, e2])
        
        d1 = self.dec1(d2, training=training)
        d1 = layers.concatenate([d1, e1])
        
        # Output
        output = self.final(d1)
        
        return output


class PatchGANDiscriminator(keras.Model):
    """
    PatchGAN Discriminator para pix2pix.
    
    Classifica patches (70x70) em vez da imagem inteira,
    permitindo feedback mais granular ao generator.
    """
    
    def __init__(self):
        super().__init__()
        
        # Estrutura: Conv2D blocks → output (1 valor por patch)
        self.conv1 = self._conv_block(64, kernel_size=4, strides=2, norm=False)
        self.conv2 = self._conv_block(128, kernel_size=4, strides=2, norm=True)
        self.conv3 = self._conv_block(256, kernel_size=4, strides=2, norm=True)
        self.conv4 = self._conv_block(512, kernel_size=4, strides=1, norm=True)
        
        # Output: classificação por patch
        self.final = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')
    
    def _conv_block(self, filters: int, kernel_size: int = 4, strides: int = 2,
                   norm: bool = True) -> keras.Sequential:
        """Bloco convolucional: Conv2D + BatchNorm + LeakyReLU"""
        block = keras.Sequential([
            layers.Conv2D(filters, kernel_size, strides, padding='same')
        ])
        if norm:
            block.add(layers.BatchNormalization())
        block.add(layers.LeakyReLU(0.2))
        return block
    
    def call(self, x, training=False):
        """Forward pass"""
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.final(x)
        
        return x


class Pix2PixGAN:
    """
    Modelo completo pix2pix para treinamento.
    
    Combina Generator + Discriminator com perdas adversariais e L1.
    """
    
    def __init__(self, image_size: int = 256, lambda_l1: float = 100.0):
        """
        Inicializa o pix2pix GAN.
        
        Args:
            image_size: Tamanho das imagens
            lambda_l1: Peso da perda L1 (padrão: 100)
        """
        self.image_size = image_size
        self.lambda_l1 = lambda_l1
        
        # Modelos
        self.generator = Pix2PixGenerator(image_size)
        self.discriminator = PatchGANDiscriminator()
        
        # Otimizadores (conforme Katafuchi & Tokunaga, 2020)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        
        # Função de perda
        self.bce_loss = keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae_loss = keras.losses.MeanAbsoluteError()
    
    def discriminator_loss(self, y_real, y_fake):
        """Perda do discriminador: classificação correta de real e fake"""
        real_loss = self.bce_loss(tf.ones_like(y_real), y_real)
        fake_loss = self.bce_loss(tf.zeros_like(y_fake), y_fake)
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, y_fake, target, generated):
        """Perda do generator: enganar discriminador + L1 com target"""
        fake_loss = self.bce_loss(tf.ones_like(y_fake), y_fake)
        l1_loss = self.mae_loss(target, generated)
        return fake_loss + self.lambda_l1 * l1_loss
    
    def train_step(self, source, target):
        """
        Um passo de treinamento (generator + discriminator).
        
        Args:
            source: Imagens de entrada (saudáveis)
            target: Imagens alvo (reconstruções esperadas)
            
        Returns:
            Dicionário com perdas
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator: reconstruir imagem
            generated = self.generator(source, training=True)
            
            # Discriminator: avaliar real e fake
            y_real = self.discriminator(target, training=True)
            y_fake = self.discriminator(generated, training=True)
            
            # Perdas
            g_loss = self.generator_loss(y_fake, target, generated)
            d_loss = self.discriminator_loss(y_real, y_fake)
        
        # Backpropagation
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'l1_loss': self.mae_loss(target, generated)
        }


if __name__ == "__main__":
    print("🔧 Testando arquitetura pix2pix...")
    
    # Criar modelos
    gan = Pix2PixGAN(image_size=256)
    
    # Teste com dummy input
    dummy_input = np.random.randn(2, 256, 256, 3).astype(np.float32)
    
    # Generator
    output_g = gan.generator(dummy_input)
    print(f"✅ Generator output shape: {output_g.shape}")
    
    # Discriminator
    output_d = gan.discriminator(dummy_input)
    print(f"✅ Discriminator output shape: {output_d.shape}")
    
    print("\n✅ Modelos criados com sucesso!")
