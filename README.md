# Diagnóstico de Anomalias em Folhas usando pix2pix GAN
## Projeto 2 - Introdução à Inteligência Artificial (UnB 2025/2)

---

## 📋 Informações do Trabalho

**Disciplina:** Introdução à Inteligência Artificial  
**Instituição:** Universidade de Brasília (UnB)  
**Período:** 2025/2  
**Data de Entrega:** Dezembro 2025

### 👥 Alunos

| Nome | Matrícula |
|------|-----------|
| Vinícius Bowen | 180079239 |
| Mateus Filho | 221000080 |
| Lucas Drummond | 231011650 |

---

## 🔗 Links Importantes

| Recurso | Link |
|---------|------|
| **Google Colab** | |
| **Repositório GitHub** | |
| **Interface Interativa** | |

---

## 📖 Resumo do Projeto

Este projeto implementa um sistema de **detecção automática de anomalias em folhas** utilizando **Redes Generativas Adversariais Condicionadas (pix2pix GAN)**. O sistema foi desenvolvido seguindo as especificações da disciplina de Inteligência Artificial, com o objetivo de identificar doenças em folhas através da análise de discrepâncias entre imagens originais e reconstruídas.

### Objetivo Principal

Diagnosticar automaticamente a presença de anomalias (doenças) em imagens de folhas, classificando-as como:
- **Saudáveis** (folhas sem doenças)
- **Doentes** (folhas com presença de anomalias)

### Metodologia

A solução utiliza uma abordagem inovadora baseada em **reconstrução de imagens**:

1. **Treino exclusivo com folhas saudáveis**: O modelo aprende os padrões normais de uma folha saudável
2. **Detecção por discrepância**: Folhas doentes apresentam desvios no padrão aprendido
3. **Mapa de anomalia**: Visualização pixel-a-pixel das regiões afetadas

---

## 🏗️ Arquitetura do Sistema

### 1. pix2pix GAN

Implementação completa do modelo **pix2pix** (Isola et al., 2017) com as seguintes componentes:

#### Generator (U-Net)

Uma arquitetura U-Net com conexões skip que aprende a mapear imagens entre domínios:

```
Encoder (Downsampling):
  Conv2D (64 filtros)   → 256×256 → 128×128
  Conv2D (128 filtros)  → 128×128 → 64×64
  Conv2D (256 filtros)  → 64×64 → 32×32
  Conv2D (512 filtros)  → 32×32 → 16×16
  Conv2D (512 filtros)  → 16×16 → 8×8
  Conv2D (512 filtros)  → 8×8 → 4×4
  Conv2D (512 filtros)  → 4×4 → 2×2
  
Bottleneck:
  Conv2D (512 filtros)  → 2×2 → 1×1

Decoder (Upsampling + Skip Connections):
  Conv2DTranspose (512) → 1×1 → 2×2 (com Dropout 0.5)
  Conv2DTranspose (512) → 2×2 → 4×4 (com Dropout 0.5)
  Conv2DTranspose (512) → 4×4 → 8×8 (com Dropout 0.5)
  Conv2DTranspose (512) → 8×8 → 16×16
  Conv2DTranspose (256) → 16×16 → 32×32
  Conv2DTranspose (128) → 32×32 → 64×64
  Conv2DTranspose (64)  → 64×64 → 128×128
  Conv2DTranspose (3)   → 128×128 → 256×256 (tanh)

Total de Parâmetros: ~54M
```

**Características:**
- Normalização em lotes (BatchNormalization) após convoluções
- Conexões skip entre camadas simétricas do encoder/decoder
- Ativação ReLU no decoder, LeakyReLU (α=0.2) no encoder
- Dropout nas primeiras 3 camadas do decoder (50%) para regularização
- Saída com ativação tanh normalizada em [-1, 1]

#### Discriminator (PatchGAN)

Um discriminador que classifica patches 70×70 para melhor captura de detalhes locais:

```
Estrutura:
  Input (256×256)
    ↓ Conv2D (64)     → 128×128 (stride 2, sem BatchNorm)
    ↓ Conv2D (128)    → 64×64 (stride 2)
    ↓ Conv2D (256)    → 32×32 (stride 2)
    ↓ Conv2D (512)    → 15×15 (stride 2)
    ↓ Conv2D (512)    → 14×14 (stride 1)
    ↓ Conv2D (1)      → 13×13 (stride 1) - Output

Total de Parâmetros: ~2.8M
```

**Características:**
- Classificação em patches para melhor granularidade
- Feedback discriminativo mais rico durante treinamento
- Ativação LeakyReLU (α=0.2) em todas as camadas

#### Função de Perda Combinada

$$\mathcal{L} = \mathcal{L}_{adv} + \lambda_{L1} \cdot \mathcal{L}_{L1}$$

Onde:
- $\mathcal{L}_{adv}$ = Perda adversarial (binary cross-entropy)
- $\mathcal{L}_{L1}$ = Distância L1 entre saída e alvo
- $\lambda_{L1}$ = 100 (peso do termo L1)

### 2. Detecção de Anomalias

Módulo que calcula índices de anomalia baseado em discrepâncias:

$$A(x,y) = ||I(x,y) - R(x,y)||^2$$

Onde:
- $I(x,y)$ = Pixel original
- $R(x,y)$ = Pixel reconstruído
- $A(x,y)$ = Índice de anomalia (0 = normal, alto = anômalo)

**Métricas de Qualidade:**
- **SSIM (Structural Similarity Index)**: Similaridade estrutural (0-1)
- **PSNR (Peak Signal-to-Noise Ratio)**: Razão sinal-ruído em dB
- **Threshold automático (Otsu)**: Binarização da anomalia

### 3. Visualização Grad-CAM

Implementação de **Gradient-weighted Class Activation Maps** para interpretabilidade:

- Visualiza quais regiões influenciam a decisão do modelo
- Utilitário GradCAMVisualizer para geração de heatmaps
- Integração com a interface interativa

### 4. Data Loader

Sistema robusto de carregamento de dados com:

- Suporte a múltiplos formatos (PNG, JPG, JPEG, TIFF)
- Redimensionamento automático para 256×256
- Normalização [-1, 1] para compatibilidade com pix2pix
- Carregamento estruturado (treino, teste saudável, teste doente)

---

## 📊 Estrutura do Projeto

```
IIA-p2/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências Python
│
├── data/                              # Dataset
│   ├── train_healthy/                 # Imagens de treino (folhas saudáveis)
│   ├── test_healthy/                  # Imagens teste (folhas saudáveis)
│   └── test_diseased/                 # Imagens teste (folhas doentes)
│
├── src/                               # Código-fonte principal
│   ├── __init__.py
│   ├── data_loader.py                 # Carregamento de dataset
│   ├── pix2pix_gan.py                 # Modelo pix2pix GAN
│   ├── anomaly_detection.py           # Cálculo de índices de anomalia
│   ├── gradcam.py                     # Visualização Grad-CAM
│   ├── utils.py                       # Utilitários (managers, visualizers)
│   └── __pycache__/
│
├── notebooks/                         # Jupyter Notebooks
│   ├── IIA_local.ipynb                # Notebook para execução local
│   └── IIA_colab.ipynb                # Notebook para Google Colab
│
├── interface/                         # Interface Interativa
│   └── app.py                         # App Streamlit
│
├── outputs/                           # Resultados e saídas
│   ├── gradcam/                       # Visualizações Grad-CAM
│   ├── anomaly_maps/                  # Mapas de anomalia
│   ├── reconstructions/               # Imagens reconstruídas
│   └── *.png                          # Gráficos e análises
│
└── models/                            # Modelos treinados (criado automaticamente)
```

---

## 🔧 Dependências

**Versões Recomendadas:**

| Pacote | Versão | Propósito |
|--------|--------|----------|
| TensorFlow | ≥2.16.0 | Deep Learning framework |
| Keras | ≥3.0.0 | API de modelos |
| OpenCV | ≥4.8.0 | Processamento de imagens |
| Pillow | ≥10.0.0 | Manipulação de imagens |
| scikit-image | ≥0.21.0 | Métricas de qualidade (SSIM, PSNR) |
| NumPy | ≥1.24.0 | Computação numérica |
| SciPy | ≥1.11.0 | Computação científica |
| Matplotlib | ≥3.7.0 | Visualização |
| Seaborn | ≥0.12.0 | Visualização estatística |
| scikit-learn | ≥1.3.0 | ML utilities (métricas ROC, etc) |
| pandas | ≥2.0.0 | Análise de dados |
| Streamlit | ≥1.28.0 | Interface interativa |
| Jupyter | ≥1.0.0 | Notebooks interativos |
| tqdm | ≥4.65.0 | Barras de progresso |

---

## 🚀 Como Executar

### Opção 1: Notebook Local (Recomendado para Desenvolvimento)

```bash
# 1. Clonar repositório
git clone <URL_REPOSITORIO>
cd IIA-p2

# 2. Criar ambiente virtual (opcional mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar Jupyter notebook
jupyter notebook notebooks/IIA_local.ipynb
```

### Opção 2: Google Colab (Recomendado para Treinamento com GPU)

```python
# No Colab, execute as seguintes células para setup:

# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar repositório
!git clone <URL_REPOSITORIO>
%cd IIA-p2

# 3. Instalar dependências
!pip install -r requirements.txt

# 4. Executar o notebook
!jupyter nbconvert --to notebook --execute notebooks/IIA_colab.ipynb
```

### Opção 3: Interface Interativa (Streamlit)

```bash
# No diretório raiz do projeto
streamlit run interface/app.py

# Acessar em: http://localhost:8501
```

---

## 📝 Workflow do Projeto

### 1. **Preparação de Dados** (`data_loader.py`)

```python
loader = DataLoader(image_size=256)
X_train, X_test_h, X_test_d, names_train, names_test_h, names_test_d = \
    loader.load_dataset('data/')

# Resultado:
# X_train: (N_train, 256, 256, 3) - folhas saudáveis para treino
# X_test_h: (N_test_h, 256, 256, 3) - folhas saudáveis para teste
# X_test_d: (N_test_d, 256, 256, 3) - folhas doentes para teste
```

### 2. **Construção do Modelo** (`pix2pix_gan.py`)

```python
gan = Pix2PixGAN(image_size=256, lambda_l1=100.0)

# Generator: ~54M parâmetros
# Discriminator: ~2.8M parâmetros
# Total: ~56.8M parâmetros
```

### 3. **Treinamento**

```python
gan.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    d_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
)

history = gan.fit(
    X_train,  # Folhas saudáveis apenas
    epochs=100,
    batch_size=8,
    validation_split=0.1
)
```

**Características do Treinamento:**
- Otimizadores Adam com learning rate 2e-4 e β₁=0.5
- Batch size: 8
- Épocas: até 100 (com early stopping recomendado)
- Validação: 10% dos dados de treino

### 4. **Detecção de Anomalias** (`anomaly_detection.py`)

```python
detector = AnomalyDetector(threshold_method='otsu')

# Para cada imagem de teste:
# 1. Reconstruir com o generator
reconstructed = gan.generator(image, training=False)

# 2. Calcular mapa de anomalia
anomaly_map, anomaly_score = detector.compute_anomaly_map(
    original=image,
    reconstructed=reconstructed,
    return_normalized=True
)

# 3. Binarizar (normal vs anômalo)
binary_map = detector.binarize_anomaly_map(anomaly_map)

# 4. Calcular métricas
metrics = {
    'ssim': anomaly_map SSIM,
    'psnr': anomaly_map PSNR,
    'anomaly_score': anomaly_score,
    'diagnosis': 'Healthy' if anomaly_score < threshold else 'Diseased'
}
```

### 5. **Visualização e Análise** (`gradcam.py`, `utils.py`)

```python
visualizer = GradCAMVisualizer(gan.generator)

# Gerar heatmap para interpretabilidade
heatmap = visualizer.generate_gradcam(image, layer_name='dec1')

# Visualizar reconstrução e anomalia lado-a-lado
visualizer.plot_reconstruction_analysis(
    original=image,
    reconstructed=reconstructed,
    anomaly_map=anomaly_map,
    diagnosis='Diseased'
)
```

---

## 📈 Resultados Esperados

### Métricas de Desempenho

O sistema fornece as seguintes métricas para cada imagem:

| Métrica | Descrição | Range |
|---------|-----------|-------|
| **Anomaly Score** | Média do mapa de anomalia | [0, 1] |
| **SSIM** | Similaridade estrutural | [0, 1] |
| **PSNR** | Razão sinal-ruído | [dB] |
| **Diagnosis** | Classificação final | Healthy/Diseased |
| **Confidence** | Confiança da predição | [0, 1] |

### Resultados por Categoria

#### Folhas Saudáveis (Teste)
- **SSIM alto** (~0.9+): Reconstrução fiel
- **PSNR alto** (>30 dB): Baixa distorção
- **Anomaly Score baixo** (<0.2): Poucos desvios

#### Folhas Doentes (Teste)
- **SSIM mais baixo** (~0.7-0.85): Discrepâncias visíveis
- **PSNR mais baixo** (20-30 dB): Maior distorção
- **Anomaly Score alto** (>0.3): Desvios significativos
- **Localização** de anomalias no mapa corresponde a lesões visuais

### Exemplos de Saída

O sistema gera para cada imagem:

1. **Reconstrução**: Versão reconstruída pelo modelo
2. **Mapa de Anomalia**: Visualização em heatmap das regiões anômalas
3. **Mapa Binarizado**: Classificação pixel-a-pixel (normal/anômalo)
4. **Grad-CAM**: Regiões que influenciam a decisão
5. **Relatório**: Métricas quantitativas

---

## 🎯 Implementação das Especificações do Projeto

Este projeto implementa completamente as especificações fornecidas:

### ✅ Componentes Obrigatórios

| Componente | Status | Detalhes |
|------------|--------|----------|
| **pix2pix GAN** | ✅ Implementado | Generator U-Net + Discriminator PatchGAN |
| **Detecção de Anomalias** | ✅ Implementado | Fórmula: A(x,y) = \|\|I(x,y) - R(x,y)\|\|² |
| **Métricas de Qualidade** | ✅ Implementado | SSIM, PSNR, Anomaly Score |
| **Visualizações** | ✅ Implementado | Mapas de anomalia, Grad-CAM |
| **Dataset Separado** | ✅ Organizado | Treino saudável, teste saudável, teste doente |
| **Notebook Jupyter** | ✅ Disponível | IIA_local.ipynb e IIA_colab.ipynb |
| **Interface Interativa** | ✅ Implementada | Streamlit app com upload de imagens |

### ✅ Componentes Bônus

| Componente | Status | Detalhes |
|------------|--------|----------|
| **Grad-CAM** | ✅ Implementado | Visualização de regiões influentes |
| **Interface Streamlit** | ✅ Implementada | App web interativa |
| **Google Colab** | ✅ Otimizado | Notebook com GPU support |
| **Métricas Avançadas** | ✅ Implementadas | ROC-AUC, Matriz de Confusão |

---

## 📚 Referências Bibliográficas

1. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017).** "Image-to-Image Translation with Conditional Adversarial Networks." In *CVPR 2017*. [arXiv:1611.05957](https://arxiv.org/abs/1611.05957)

2. **Katafuchi, K., & Tokunaga, M. (2020).** "Unsupervised Anomaly Detection on Optical Network Data using Generative Adversarial Network." In *NOMS 2020*. IEEE.

3. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014).** "Generative Adversarial Nets." In *NIPS 2014*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

4. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** "U-Net: Convolutional Networks for Biomedical Image Segmentation." In *MICCAI 2015*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

5. **Selvaraju, R. R., Coignard, A., Das, A., et al. (2016).** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." [arXiv:1610.02055](https://arxiv.org/abs/1610.02055)

---

## 💡 Notas Técnicas

### Normalização de Imagens

- **Entrada**: Imagens BGR em [0, 255] (formato OpenCV)
- **Processamento**: Conversão RGB e redimensionamento 256×256
- **Normalização**: $(I / 127.5) - 1.0$ → [-1, 1] (compatível com tanh)
- **Visualização**: Desnormalização $(I + 1.0) / 2.0$ → [0, 1]

### Treinamento

- **Dataset de Treino**: **Apenas folhas saudáveis**
  - O modelo aprende a reconstruir características normais
  - Folhas doentes terão reconstruções com discrepâncias
  
- **Otimização**:
  - Adam optimizer: learning_rate=2e-4, β₁=0.5, β₂=0.999
  - Batch normalization após convoluções (exceto primeira camada discriminator)
  - Dropout 50% nas primeiras 3 camadas do decoder

### Avaliação

A detecção é baseada no princípio de **anomalia por reconstrução**:

1. **Imagem Saudável**: Reconstrução fiel → SSIM alto, PSNR alto
2. **Imagem Doente**: Reconstrução com erro → SSIM baixo, PSNR baixo

O modelo nunca viu folhas doentes no treino, então as discrepâncias no teste indicam anomalias.

---

## 🐛 Troubleshooting

### Problema: Out of Memory (OOM)

**Solução:**
```python
# Reduzir batch size
batch_size = 4  # Ao invés de 8

# Reduzir image size (não recomendado)
image_size = 128

# Usar mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Problema: Modelo não convergindo

**Solução:**
```python
# Verificar learning rate
learning_rate = 1e-4  # Mais baixo para convergência estável

# Verificar balanço G-D (devem ter perdas similares)
# Se G loss >> D loss: Aumentar lambda_l1 para 50
gan = Pix2PixGAN(image_size=256, lambda_l1=50.0)
```

### Problema: GPU não detectada

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Se vazio: TensorFlow não encontrou GPU

# Verificar instalação CUDA/cuDNN
!nvidia-smi  # Verificar drivers
```

---

## 📄 Licença

Este projeto é desenvolvido para fins educacionais na disciplina de Introdução à Inteligência Artificial - UnB 2025/2.

---

## 📞 Contato e Suporte

Para dúvidas sobre o projeto:
- Vinícius Bowen: 180079239
- Mateus Filho: 221000080
- Lucas Drummond: 231011650

**Data de Compilação:** Dezembro 2025

---

**Última Atualização:** Dezembro 2025
