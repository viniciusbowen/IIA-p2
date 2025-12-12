"""
Interface Interativa para Diagnóstico de Anomalias em Folhas
Projeto 2 - Introdução à Inteligência Artificial (UnB 2025/2)

Streamlit App para detecção de anomalias usando pix2pix GAN
"""

import os
import sys

# Suprimir mensagens informativas do TensorFlow/oneDNN (deve vir antes de importar tf)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

import warnings
import logging

# Suprimir warnings de deprecação do TensorFlow
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import tensorflow as tf

# Suprimir warnings adicionais do TensorFlow v1
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Adicionar src ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pix2pix_gan import Pix2PixGAN, Pix2PixGenerator
from anomaly_detection import AnomalyDetector
from gradcam import GradCAMVisualizer

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico de Anomalias em Folhas",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4caf50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# Variáveis globais para os modelos (evitar conflito Streamlit/Keras)
_generator = None
_discriminator = None
_model_loaded = False
_load_error = None

def get_models():
    """Retorna os modelos carregados, carregando-os se necessário."""
    global _generator, _discriminator, _model_loaded, _load_error
    
    if _model_loaded:
        return _generator, _discriminator, _load_error
    
    try:
        # Determinar caminho absoluto dos modelos
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(project_dir, 'models')
        
        # Tentar diferentes nomes de arquivo para o generator
        generator_names = [
            'generator_weights.weights.h5',
            'generator_weights.h5',
            'generator.h5',
        ]
        
        generator_path = None
        for name in generator_names:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                generator_path = path
                break
        
        if generator_path is None:
            _load_error = f"Nenhum arquivo de modelo encontrado em {models_dir}"
            _model_loaded = True
            return None, None, _load_error
        
        # Criar instância do modelo e carregar pesos
        gan = Pix2PixGAN(image_size=256, lambda_l1=100.0)
        
        # Construir modelos passando um tensor de exemplo
        dummy_input = tf.zeros((1, 256, 256, 3))
        _ = gan.generator(dummy_input, training=False)
        _ = gan.discriminator(dummy_input, training=False)
        
        # Carregar pesos do generator
        gan.generator.load_weights(generator_path)
        _generator = gan.generator
        
        # Tentar carregar o discriminator para Grad-CAM
        discriminator_names = [
            'discriminator_weights.weights.h5',
            'discriminator_weights.h5',
            'discriminator.h5',
        ]
        
        for name in discriminator_names:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                gan.discriminator.load_weights(path)
                _discriminator = gan.discriminator
                break
        
        _model_loaded = True
        return _generator, _discriminator, None
        
    except Exception as e:
        import traceback
        _load_error = f"Erro ao carregar modelo: {e}\n{traceback.format_exc()}"
        _model_loaded = True
        return None, None, _load_error

def preprocess_image(image, target_size=256):
    """Pré-processa imagem para entrada no modelo"""
    # Converter para RGB (modelo notebook treina em RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize((target_size, target_size))
    
    # Normalizar para [-1, 1] (padrão pix2pix com tanh)
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

def main():
    # Header
    st.markdown('<div class="main-header">🍃 Diagnóstico de Anomalias em Folhas</div>', unsafe_allow_html=True)
    st.markdown("### Detecção de Doenças usando pix2pix GAN")
    st.markdown("**Projeto 2 - Introdução à Inteligência Artificial (UnB 2025/2)**")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2e7d32/ffffff?text=UnB+IIA", width='stretch')
        st.markdown("## ⚙️ Configurações")
        
        show_gradcam = st.checkbox("Exibir Grad-CAM", value=True)
        colormap_choice = st.selectbox(
            "Mapa de Cores para Anomalia",
            ["jet", "hot", "viridis", "plasma", "inferno"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Informações do Modelo")
        st.info("""
        **Arquitetura**: pix2pix GAN
        - **Generator**: U-Net 8 níveis
        - **Discriminator**: PatchGAN 70×70
        - **Treinado**: 50 imagens saudáveis
        - **Métrica**: Anomaly Index = ||I - R||²
        """)
        
        # Limiar fixo
        anomaly_threshold = 0.0005
        
        st.markdown("---")
        st.markdown("### 📚 Referências")
        st.markdown("""
        - Isola et al. (2017) - pix2pix
        - Katafuchi & Tokunaga (2020)
        """)
    
    # Carregar modelo (usando variáveis globais para evitar conflito Streamlit/Keras)
    generator, discriminator, error_msg = get_models()
    
    if generator is None:
        st.error("⚠️ Modelo não encontrado. Execute o notebook de treinamento primeiro!")
        if error_msg:
            st.error(error_msg)
        return
    
    st.success("✅ Modelo carregado com sucesso!")
    if discriminator is not None:
        st.info("✅ Grad-CAM disponível")
    
    detector = AnomalyDetector()
    
    # Upload de imagem
    st.markdown("## 📤 Upload de Imagem")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Selecione uma imagem de folha (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Faça upload de uma imagem de folha para análise"
        )
    
    with col2:
        st.markdown("#### Exemplos:")
        example_choice = st.radio(
            "Ou use um exemplo:",
            ["Nenhum", "Saudável 1", "Saudável 2", "Doente 1", "Doente 2"]
        )
    
    # Processar imagem
    if uploaded_file is not None or example_choice != "Nenhum":
        st.divider()
        st.markdown("## 🔍 Análise")
        
        # Carregar imagem
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            # Carregar exemplo
            example_paths = {
                "Saudável 1": "../data/test_healthy/00001.jpg",
                "Saudável 2": "../data/test_healthy/00010.jpg",
                "Doente 1": "../data/test_diseased/00001.jpg",
                "Doente 2": "../data/test_diseased/00050.jpg"
            }
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(os.path.dirname(script_dir), 'data')
                # Atualizar paths com nomes reais dos arquivos
                example_paths = {
                    "Saudável 1": os.path.join(data_dir, "test_healthy", "leaf a1-a3 ab_0.jpg"),
                    "Saudável 2": os.path.join(data_dir, "test_healthy", "leaf a10-a12 ab_0.jpg"),
                    "Doente 1": os.path.join(data_dir, "test_diseased", "a976-979 ab_2.jpg"),
                    "Doente 2": os.path.join(data_dir, "test_diseased", "a1001-1003 ab_0.jpg")
                }
                image = Image.open(example_paths.get(example_choice, list(example_paths.values())[0]))
            except Exception as ex:
                st.warning(f"⚠️ Exemplo não encontrado: {ex}. Use o upload de arquivo.")
                return
        
        # Pré-processar
        img_array = preprocess_image(image)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Gerar reconstrução
        with st.spinner("🔄 Reconstruindo imagem..."):
            reconstructed = generator(img_batch, training=False).numpy()[0]
            
            # Desnormalizar de [-1, 1] para [0, 1]
            reconstructed = (reconstructed + 1.0) / 2.0
            reconstructed = np.clip(reconstructed, 0, 1)
            
            # O modelo gera RGB (3 canais), então não precisa converter
            # Apenas converter o original grayscale para RGB para comparação
            img_array_normalized = (img_array + 1.0) / 2.0
            
            # Converter original grayscale para RGB (repetir canal)
            if img_array_normalized.shape[-1] == 1:
                img_array_rgb = np.repeat(img_array_normalized, 3, axis=-1)
            else:
                img_array_rgb = img_array_normalized
        
        # DEBUG: Verificar se a reconstrução está funcionando
        with st.expander("🔍 Debug - Informações Técnicas"):
            st.write(f"**Shape Original:** {img_array_rgb.shape}")
            st.write(f"**Shape Reconstruída:** {reconstructed.shape}")
            st.write(f"**Range Original:** [{img_array_rgb.min():.3f}, {img_array_rgb.max():.3f}]")
            st.write(f"**Range Reconstruída:** [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            diff_abs = np.mean(np.abs(img_array_rgb - reconstructed))
            diff_squared = np.mean((img_array_rgb - reconstructed) ** 2)
            st.write(f"**Diferença média absoluta (MAE):** {diff_abs:.6f}")
            st.write(f"**Diferença média quadrática (MSE):** {diff_squared:.6f}")
        
        # Calcular anomalia (usar versões RGB para métricas)
        with st.spinner("📊 Calculando métricas..."):
            anomaly_map, anomaly_score = detector.compute_anomaly_map(img_array_rgb, reconstructed)
            psnr = detector.compute_psnr(img_array_rgb, reconstructed)
            ssim = detector.compute_ssim(img_array_rgb, reconstructed)
            
            # Colormap
            anomaly_colored = detector.colormap_anomaly(anomaly_map, cmap=colormap_choice)
        
        # Grad-CAM (opcional)
        if show_gradcam and discriminator is not None:
            with st.spinner("🔥 Aplicando Grad-CAM..."):
                try:
                    gradcam_viz = GradCAMVisualizer(discriminator)
                    heatmap = gradcam_viz.compute_gradcam(img_batch)[0]
                    gradcam_overlay = gradcam_viz.overlay_gradcam(img_array, heatmap)
                except Exception as e:
                    st.warning(f"⚠️ Erro ao calcular Grad-CAM: {e}")
                    show_gradcam = False
        elif show_gradcam and discriminator is None:
            st.warning("⚠️ Grad-CAM não disponível: modelo discriminator não encontrado")
            show_gradcam = False
        
        # Exibir métricas
        st.markdown("### 📈 Métricas de Qualidade")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            anomaly_status = '🔴 ALTA ANOMALIA' if anomaly_score > anomaly_threshold else '🟢 BAIXA ANOMALIA'
            st.markdown(f"""
            <div class="metric-box">
                <h3>📊 Anomaly Score</h3>
                <h1>{anomaly_score:.6f}</h1>
                <p>{anomaly_status}</p>
                <small>Limiar: {anomaly_threshold:.3f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>📡 PSNR</h3>
                <h1>{psnr:.2f} dB</h1>
                <p>Peak Signal-to-Noise Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>🎯 SSIM</h3>
                <h1>{ssim:.4f}</h1>
                <p>Structural Similarity Index</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizações
        st.markdown("### 🖼️ Visualizações")
        
        if show_gradcam:
            col1, col2, col3, col4 = st.columns(4)
            titles = ["Original", "Reconstrução", "Mapa de Anomalia", "Grad-CAM"]
            images = [img_array_rgb, reconstructed, anomaly_colored, gradcam_overlay]
        else:
            col1, col2, col3 = st.columns(3)
            titles = ["Original", "Reconstrução", "Mapa de Anomalia"]
            images = [img_array_rgb, reconstructed, anomaly_colored]
        
        cols = [col1, col2, col3, col4] if show_gradcam else [col1, col2, col3]
        
        for col, title, img in zip(cols, titles, images):
            with col:
                st.markdown(f"**{title}**")
                st.image(img, width='stretch')
        
        # Análise detalhada
        with st.expander("🔬 Análise Detalhada"):
            st.markdown("#### Interpretação dos Resultados:")
            
            st.markdown(f"""
            **Anomaly Score ({anomaly_score:.6f})**:
            - Baseado na fórmula: $A(x,y) = ||I(x,y) - R(x,y)||^2$
            - Valores típicos:
              - Saudáveis: < 0.002
              - Doentes: > 0.003
            
            **PSNR ({psnr:.2f} dB)**:
            - Mede qualidade da reconstrução
            - Valores maiores = melhor reconstrução
            - Folhas saudáveis tendem a ter PSNR > 25 dB
            
            **SSIM ({ssim:.4f})**:
            - Similaridade estrutural (0 a 1)
            - Valores maiores = mais similar ao original
            - Folhas saudáveis: SSIM > 0.85
            """)
            
            if show_gradcam:
                st.markdown("""
                **Grad-CAM**:
                - Áreas vermelhas = alta ativação do discriminador
                - Indica regiões que o modelo considera "suspeitas"
                - Em folhas doentes, destaca lesões e manchas
                """)
        
        # Download de resultados
        st.markdown("### 💾 Download de Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Salvar mapa de anomalia
            anomaly_img = Image.fromarray((anomaly_colored * 255).astype(np.uint8))
            from io import BytesIO
            buf = BytesIO()
            anomaly_img.save(buf, format='PNG')
            st.download_button(
                "📥 Baixar Mapa de Anomalia",
                data=buf.getvalue(),
                file_name="anomaly_map.png",
                mime="image/png"
            )
        
        with col2:
            # Salvar reconstrução
            recon_img = Image.fromarray((reconstructed * 255).astype(np.uint8))
            buf2 = BytesIO()
            recon_img.save(buf2, format='PNG')
            st.download_button(
                "📥 Baixar Reconstrução",
                data=buf2.getvalue(),
                file_name="reconstruction.png",
                mime="image/png"
            )
        
        with col3:
            # Relatório CSV
            diagnosis = 'Doente' if anomaly_score > anomaly_threshold else 'Saudável'
            report = f"""Imagem,Anomaly Score,PSNR,SSIM,Limiar,Diagnóstico
{uploaded_file.name if uploaded_file else example_choice},{anomaly_score:.6f},{psnr:.2f},{ssim:.4f},{anomaly_threshold:.3f},{diagnosis}
"""
            st.download_button(
                "📥 Baixar Relatório (CSV)",
                data=report,
                file_name="diagnostico.csv",
                mime="text/csv"
            )
    
    else:
        # Instruções
        st.info("""
        ### 👋 Bem-vindo!
        
        Esta interface permite diagnosticar doenças em folhas usando inteligência artificial.
        
        **Como usar**:
        1. Faça upload de uma imagem de folha ou escolha um exemplo
        2. Aguarde o processamento (alguns segundos)
        3. Visualize os resultados e o diagnóstico
        4. Baixe os mapas de anomalia e relatórios
        
        **Funcionalidades**:
        - ✅ Reconstrução usando pix2pix GAN
        - ✅ Cálculo de métricas (PSNR, SSIM, Anomaly Score)
        - ✅ Mapas de anomalia coloridos
        - ✅ Grad-CAM para interpretabilidade
        - ✅ Download de resultados
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Projeto 2 - Introdução à Inteligência Artificial (UnB 2025/2)</strong></p>
        <p>Baseado em: Katafuchi & Tokunaga (2020) + Isola et al. (2017)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
