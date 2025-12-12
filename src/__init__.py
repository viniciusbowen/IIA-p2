"""
Projeto 2 - IA: Diagnóstico de Anomalias em Folhas com pix2pix

Módulo principal que orquestra todo o pipeline de treinamento e avaliação.

Autor: Sistema Automático
Data: Dezembro 2025
"""

from data_loader import DataLoader
from pix2pix_gan import Pix2PixGAN
from anomaly_detection import AnomalyDetector
from gradcam import GradCAMVisualizer, InterpretabilityVisualizer
from utils import ModelManager, ImageVisualizer, DataProcessor

__version__ = "1.0.0"
__all__ = [
    'DataLoader',
    'Pix2PixGAN',
    'AnomalyDetector',
    'GradCAMVisualizer',
    'InterpretabilityVisualizer',
    'ModelManager',
    'ImageVisualizer',
    'DataProcessor',
]
