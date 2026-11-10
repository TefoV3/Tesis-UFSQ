# Predicción temprana de estrés/ansiedad con datos fisiológicos multimodales

**Autor:** Estéfano Galarza
**Tema de tesis:** Predicción temprana de estrés usando datos fisiológicos multimodales
**Notebook principal:** `modelo_ordenadov3CUDA_corregido.ipynb` (kernel `TesisCUDA`, Python 3.12)

Este repositorio contiene el análisis completo de la tesis: exploración de datos, robustez al **ruido gaussiano**, selección de características, optimización de un modelo base (Random Forest) y, con mayor énfasis, su comparación contra **8 Modelos Fundacionales Tabulares basados en Transformers (TFM)** en un esquema de inferencia *zero-shot / in-context learning*, todo acelerado en GPU (CUDA).

---

## Tabla de contenidos

1. [Resumen](#1-resumen)
2. [Datos](#2-datos)
3. [Metodología](#3-metodología)
4. [Modelos evaluados](#4-modelos-evaluados)
5. [Resultados](#5-resultados)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Requisitos](#7-requisitos)
8. [Instalación del entorno (Conda + CUDA)](#8-instalación-del-entorno-conda--cuda)
9. [Cómo ejecutar el notebook](#9-cómo-ejecutar-el-notebook)
10. [Archivos de salida](#10-archivos-de-salida)
11. [Solución de problemas](#11-solución-de-problemas)
12. [Reproducibilidad y licencias](#12-reproducibilidad-y-licencias)

---

## 1. Resumen

Los trastornos de ansiedad son un problema global de salud mental, y los métodos tradicionales de detección (cuestionarios de autoreporte) no son dinámicos ni cuantitativos. La tecnología *wearable* permite monitorear en tiempo real la respuesta del sistema nervioso autónomo mediante señales como la actividad electrodérmica (EDA), el volumen de pulso sanguíneo (BVP) y el electrocardiograma (ECG).

**Objetivo:** evaluar el desempeño de Modelos Fundacionales Tabulares basados en Transformers para clasificar estados fisiológicos (reposo, emoción positiva y estrés), contrastándolos con un modelo base robusto (Random Forest optimizado), y cuantificar la degradación del modelo base al inyectar ruido gaussiano sintético con distintos niveles de relación señal-ruido (SNR). Se usan Accuracy, F1-Score, curvas ROC y AUC, con validación cruzada (por sujeto en la comparación final) para evitar sobreajuste.

**Alcance:** el análisis de ruido se aplicó al Random Forest (118 variables); la comparación entre Transformers se realizó con datos sin ruido (18 variables). Evaluar los Transformers bajo ruido queda como trabajo futuro.

El proyecto sigue una estructura tipo CRISP-DM: Business Understanding → Data Understanding/EDA → Data Preparation → Modeling → Evaluation → Deployment.

## 2. Datos

| Característica | Valor |
|---|---|
| Fuente | WESAD preprocesado (`may14_feats4.csv`) |
| Ruta esperada por el notebook | `../anxietyFB-main/data/may14_feats4.csv` (relativa al notebook) |
| Instancias | 577 |
| Columnas | 120 (118 predictoras + `label` + `subject`) |
| Valores nulos | 0 (integridad del 100 %) |
| Clases | 3 (`0`, `1`, `2`), ver tabla siguiente |
| Distribución original | clase 1: 54.07 %, clase 2: 30.33 %, clase 0: 15.60 % |
| Tras *undersampling* | 270 muestras (90 por clase) |

**Definición de las clases:**

| Clase | Estado | Descripción |
|---|---|---|
| `0` | Reposo / neutral (línea base) | Periodo prolongado (aprox. 20 minutos) de actividades tranquilas, como leer revistas, para establecer una línea base fisiológica sin alteración emocional. |
| `1` | Emoción positiva inducida | Visualización de videoclips graciosos y neutrales, diseñados para provocar emociones positivas o de alegría. |
| `2` | Estrés psicosocial (alta activación simpática) | Inducido con pruebas estandarizadas de laboratorio como el Trier Social Stress Test (TSST): hablar en público y aritmética mental bajo presión. |

El notebook trata el problema como **clasificación multiclase de 3 clases**. En la tarea binaria final (ansioso vs. no ansioso), la clase 1 suele agruparse con la clase 0 bajo "no ansioso / no estresado"; esa variante no se evalúa aquí.

> **Importante:** el archivo de datos **no** se incluye en este repositorio. Colócalo en la ruta indicada o modifica la celda 3.1 (`pd.read_csv(...)`).

## 3. Metodología

1. **EDA:** calidad estructural, distribución de clases y correlaciones (multicolinealidad entre variables derivadas de una misma señal).
2. **Pipeline A: robustez al ruido (118 variables).** Estandarización + *Noise Engine* que inyecta ruido gaussiano según la relación señal-ruido (SNR): `varianza_ruido = E[S²] / SNR`. Niveles evaluados: `SNR = [0.01, 0.1, 0.15, 0.4, 0.6]` más el *Baseline* sin ruido. Random Forest (100 árboles, `class_weight='balanced'`) con `StratifiedKFold(5)`.
3. **Pipeline B: selección de características.** *Undersampling* (`RandomUnderSampler`, `random_state=42`) → estandarización → ranking con **ReliefF** → selección de las **18 mejores variables** por el método del codo (reducción de 118 a 18, un 85 %).
4. **Random Forest optimizado.** `GridSearchCV` (81 combinaciones × 5 folds = 405 ajustes) maximizando F1 ponderado, con **`StratifiedGroupKFold(5)`** para que ningún sujeto aparezca a la vez en entrenamiento y prueba (evita fuga de información entre pacientes).
5. **Modelos fundacionales tabulares.** Evaluados con los mismos 5 folds agrupados por sujeto sobre las 18 variables, en GPU.
6. **Evaluación:** Accuracy, Precision/Recall/F1 (ponderado y macro), ROC-AUC (One-vs-Rest ponderado), Cohen's Kappa, MCC, métricas por clase, matrices de confusión, rendimiento por fold y predicciones *out-of-fold*.

**Las 18 variables seleccionadas por ReliefF:** `EDA_SCL_max`, `EDA_SCL_mean`, `EDA_SCL_min`, `EDA_SCL_C_max`, `EDA_SCL_C_mean`, `EDA_SCL_C_min`, `ECG_bpm`, `ECG_ibi`, `Resp_C_rate`, `Resp_C_Exhal_mean`, `ECG_pnn20`, `Resp_C_Inhal_std`, `net_acc_max`, `Resp_C_Exhal_std`, `Resp_C_Inhal_mean`, `ECG_std`, `TEMP_max`, `TEMP_mean`.

**Mejores hiperparámetros del Random Forest:** `n_estimators=200`, `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=4` (F1 ponderado medio en validación cruzada: 0.8692).

## 4. Modelos evaluados

| Modelo | Familia / enfoque | Paquete |
|---|---|---|
| Random Forest (optimizado) | Ensamble de árboles (tradicional) | `scikit-learn` |
| TabICL (v1) | TFM – prior sintético | `tabicl` |
| TabICL (v2) | TFM – prior sintético mejorado | `tabicl` |
| Mitra | TFM – prior mixto | `autogluon.tabular` |
| TabDPT | TFM – preentrenado en datos reales | `tabdpt` |
| TabPFN | TFM – prior sintético | `tabpfn` |
| TabFM | TFM – modelo fundacional (Google Research) | `tabfm` |
| iLTM | TFM – integrado (árboles + MLP) | `iltm` |
| EXAONE-Tabular | TFM – CAST (LG AI Research) | `exaonetabular` |

## 5. Resultados

**Robustez al ruido (Random Forest, 118 variables, `StratifiedKFold`):**

| SNR | Accuracy | F1 (ponderado) | AUC (OvR) |
|---|---|---|---|
| Baseline (sin ruido) | 0.9688 | 0.9689 | 0.9981 |
| 0.6 | 0.8023 | 0.7673 | 0.8965 |
| 0.4 | 0.7435 | 0.6906 | 0.8545 |
| 0.15 | 0.6586 | 0.6234 | 0.7832 |
| 0.1 | 0.6291 | 0.5786 | 0.7367 |
| 0.01 | 0.5061 | 0.4481 | 0.5209 |

**Comparativa final (18 variables, `StratifiedGroupKFold(5)`, 270 muestras):**

| # | Modelo | Accuracy | F1 (pond.) | ROC-AUC (OvR) | Kappa | MCC |
|---|---|---|---|---|---|---|
| 1 | Random Forest (optimizado) | 0.8741 | 0.8705 | 0.9530 | 0.8111 | 0.8188 |
| 2 | TabICL (v2) | 0.8370 | 0.8345 | 0.9493 | 0.7556 | 0.7563 |
| 3 | TabDPT | 0.8296 | 0.8248 | 0.9518 | 0.7444 | 0.7482 |
| 4 | TabICL (v1) | 0.8259 | 0.8238 | 0.9460 | 0.7389 | 0.7406 |
| 5 | iLTM | 0.8222 | 0.8198 | **0.9546** | 0.7333 | 0.7341 |
| 6 | EXAONE-Tabular | 0.8222 | 0.8185 | 0.9434 | 0.7333 | 0.7352 |
| 7 | Mitra | 0.8074 | 0.8064 | 0.9299 | 0.7111 | 0.7120 |
| 8 | TabFM | 0.8037 | 0.7962 | 0.9388 | 0.7056 | 0.7115 |
| 9 | TabPFN | 0.7926 | 0.7890 | 0.9330 | 0.6889 | 0.6910 |

El Random Forest optimizado obtiene la mayor Accuracy y F1; iLTM logra el AUC más alto por una diferencia mínima (0.0015). Todos los modelos reconocen bien la clase 2 (estrés, recall de 0.93 a 0.99); los errores se concentran entre el reposo (clase 0) y la emoción positiva (clase 1). El RF tiene su menor recall en la clase 1 (0.69) y siete de los ocho TFM en la clase 0 (Mitra, en la clase 1). Los resultados completos están en `Resultados_Comparativa_9_Modelos_Tesis.xlsx`.

## 6. Estructura del repositorio

```
.
Tesis-UFSQ/
│
├── anxietyFB-main/                      # Directorio principal del pipeline base y datos
│   └── data/
│       └── may14_feats4.csv             # Dataset multimodal de bioseñales fisiológicas (WESAD procesado)
│
├── docs/                                # Documentación, reportes y recursos gráficos para la tesis
│   ├── ContextoTrabajo.pdf              # Documento base de contexto y fundamentación
│   ├── ModelosTransformers.pdf          # Referencia teórica sobre Modelos Fundacionales Tabulares (TFM)
│   ├── Presentacion_Estres_Fisiologico.pptx # Diapositivas de exposición preeliminar para la tesis
│   ├── Grafico_Barras_9_Modelos.png     # Panel comparativo gráfico de los 9 modelos evaluados
│   ├── Matrices_Confusion_9_Modelos.png # Panel visual de matrices de confusión (3x3)
│   └── Resultados_Comparativa_9_Modelos_Tesis.xlsx # Consolidado completo en Excel (Ranking, Folds, OOF)
│
├── Modelado/                            # Jupyter Notebooks de experimentación y ejecución
│   ├── modelo_ordenado_v2.2CPU.ipynb    # Script iterativo previo ejecutado en CPU
│   ├── modelo_ordenadov3.2CUDA.ipynb    # Script consolidado principal ejecutado en GPU (CUDA)
│   └── faulthandler_tabpfn.log          # Registro de eventos/depuración de TabPFN
│
├── modelosgit/                          # Repositorios externos clonados e integrados
│   ├── LimiX/                           # Módulos y pesos de LimiX (excluido en reporte final)
│   └── tabfm/                           # Implementación y utilidades del Foundation Model TabFM
│
└── README.md                            # Descripción general del repositorio y del proyecto de tesis
```

### Estructura del notebook

El notebook incluye un índice con enlaces en su celda inicial. Resumen:

| Sección | Contenido |
|---|---|
| 1. Business Understanding | Problema, objetivo, dataset y definición de las clases |
| 2. Configuración inicial | Librerías, variables de entorno y configuración de GPU (CUDA) |
| 3. Data Understanding / EDA | Carga, calidad estructural, distribución de clases, correlaciones y hallazgos |
| 4. Data Preparation | Pipeline A (ruido gaussiano, SNR) y Pipeline B (ReliefF → 18 variables) |
| 5. Modeling | 5.1 RF con ruido · 5.2 RF optimizado · 5.3 TabICL v1 · 5.4 TabICL v2 · 5.5 Mitra · 5.6 Matrices NumPy · 5.7 TabDPT · 5.8 TabPFN · 5.9 TabFM · 5.10 iLTM · 5.11 EXAONE-Tabular |
| 6. Evaluation | Comparativa, matrices de confusión, exportación a Excel e interpretación |
| 7. Deployment | Artefactos, conclusiones, limitaciones y trabajo futuro |

## 7. Requisitos

**Hardware (usado en la tesis)**
- GPU NVIDIA GeForce RTX 5060 Desktop/Laptop (arquitectura Blackwell, `sm_120`), 7.9 GB de VRAM.
- Una GPU NVIDIA es muy recomendable; sin ella el código cae a CPU (`DEVICE = "cpu"`), pero será mucho más lento.

**Software**
- Windows 10/11 (el notebook incluye ajustes específicos para Windows).
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- `git` instalado (necesario para instalar EXAONE-Tabular desde GitHub).
- Driver NVIDIA actualizado. **No** hace falta instalar el CUDA Toolkit por separado: PyTorch trae sus propias librerías CUDA.
- Python **3.12** y **PyTorch 2.11.0 compilado con CUDA 12.8 (`cu128`)**. Las GPU Blackwell (RTX 50xx) requieren CUDA 12.8 o superior.
- Conexión a internet en la primera ejecución: los pesos de los modelos se descargan automáticamente desde Hugging Face.

## 8. Instalación del entorno (Conda + CUDA)

Abre **Anaconda Prompt** y ejecuta los comandos en orden.

### 8.1 Crear y activar el entorno

```bat
conda create -n TesisCUDA python=3.12 -y
conda activate TesisCUDA
python -m pip install --upgrade pip
```

### 8.2 Librerías base (ciencia de datos y Jupyter)

```bat
pip install numpy pandas matplotlib seaborn scikit-learn joblib openpyxl
pip install imbalanced-learn skrebate
pip install jupyter ipykernel ipywidgets tqdm
```

> `imbalanced-learn` provee el módulo `imblearn` y `skrebate` provee `ReliefF`.
> Si `pip install skrebate` falla, instálalo desde el código fuente:
> `pip install git+https://github.com/EpistasisLab/scikit-rebate.git`

### 8.3 Modelos fundacionales disponibles en PyPI

```bat
pip install tabicl tabdpt tabpfn iltm huggingface_hub
pip install "autogluon.tabular[mitra]"
pip install "tabfm[pytorch]"
```

> `autogluon.tabular[mitra]` aporta Mitra. Si el extra `[mitra]` da error, usa `pip install autogluon.tabular`.
> `tabfm[pytorch]` instala TabFM (Google Research) con el backend de PyTorch. Alternativa desde el código fuente: `git clone https://github.com/google-research/tabfm.git`, luego `cd tabfm` y `pip install -e .[pytorch]`.

### 8.4 EXAONE-Tabular (desde GitHub)

```bat
pip install git+https://github.com/LGAI-Research/EXAONE-Tabular.git
```

### 8.5 PyTorch con CUDA 12.8 (siempre al final)

Algunos paquetes anteriores pueden reemplazar PyTorch por la versión solo CPU, por eso se instala al final:

```bat
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Para reproducir exactamente la versión usada en la tesis: `pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128`.

### 8.6 Registrar el kernel de Jupyter

```bat
python -m ipykernel install --user --name TesisCUDA --display-name "TesisCUDA"
```

### 8.7 Verificar la instalación

```bat
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Salida esperada: `2.11.0+cu128 12.8 True <nombre de tu GPU>`.

Verificación de los paquetes del proyecto:

```bat
python -c "from skrebate import ReliefF; from imblearn.under_sampling import RandomUnderSampler; from tabicl import TabICLClassifier; from tabdpt import TabDPTClassifier; from tabpfn import TabPFNClassifier; from tabfm import TabFMClassifier; from iltm import iLTMClassifier; from exaonetabular import EXAONETabularClassifier; from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier; print('Todo OK')"
```

### 8.8 (Opcional) Guardar las versiones exactas

Una vez que todo funcione, congela las versiones para poder recrear el entorno idéntico:

```bat
pip freeze > requirements_TesisCUDA.txt
```

Para recrearlo después: `pip install -r requirements_TesisCUDA.txt` (instala PyTorch aparte con el comando de la sección 8.5).

### 8.9 Token de TabPFN

TabPFN requiere un token personal (se obtiene en https://ux.priorlabs.ai/account). El notebook lo busca en `~/.tabpfn/token` o en la variable de entorno `TABPFN_TOKEN`. En Anaconda Prompt puedes definirlo así (reemplaza el valor y reinicia Jupyter):

```bat
setx TABPFN_TOKEN "TU_TOKEN_AQUI"
```

> **Nunca** pegues el token dentro del notebook ni lo subas a un repositorio.

### Resumen de dependencias

| Propósito | Paquetes |
|---|---|
| Datos y gráficos | `numpy`, `pandas`, `matplotlib`, `seaborn` |
| ML clásico y métricas | `scikit-learn`, `imbalanced-learn`, `skrebate`, `joblib` |
| Exportación a Excel | `openpyxl` |
| Notebook | `jupyter`, `ipykernel`, `ipywidgets`, `tqdm` |
| Deep learning / GPU | `torch`, `torchvision` (CUDA 12.8) |
| Modelos fundacionales | `tabicl`, `tabdpt`, `tabpfn`, `tabfm`, `iltm`, `exaonetabular`, `autogluon.tabular[mitra]`, `huggingface_hub` |

## 9. Cómo ejecutar el notebook

1. Coloca el dataset en `../anxietyFB-main/data/may14_feats4.csv` (relativo al notebook).
2. Activa el entorno y abre Jupyter:

   ```bat
   conda activate TesisCUDA
   jupyter notebook
   ```
3. Abre `modelo_ordenadov3CUDA_corregido.ipynb` y selecciona el kernel **TesisCUDA**.
4. Ejecuta las celdas en orden (**Kernel → Restart & Run All**). La sección 2.2 detecta la GPU y define `DEVICE`; debes ver algo como `PyTorch 2.11.0+cu128 | compilado con CUDA: 12.8` y `Dispositivo activo para los modelos: CUDA`.

**Tiempos de inferencia aproximados** (RTX 5060 Laptop, 5 folds): TabICL v2 ≈ 4 s, TabDPT ≈ 8 s, EXAONE-Tabular ≈ 8 s, Mitra ≈ 10 s, TabICL v1 ≈ 24 s, TabFM ≈ 46 s, iLTM ≈ 63 s. El `GridSearchCV` del Random Forest (405 ajustes) es la parte más larga.

## 10. Archivos de salida

`Resultados_Comparativa_9_Modelos_Tesis.xlsx` contiene:

| Hoja | Contenido |
|---|---|
| `Ranking_Global` | Accuracy, Precision, Recall, F1 (ponderado y macro), ROC-AUC, Kappa, MCC y brecha frente al mejor |
| `Metricas_Por_Clase` | Precision, Recall, F1 y AUC por clase y modelo |
| `Matrices_de_Confusion` | Conteos y porcentajes por clase real |
| `Rendimiento_Por_Fold_CV` | Métricas por fold, media y desviación estándar |
| `Predicciones_OOF` | Predicción *out-of-fold* y acierto de cada modelo para cada ventana |

Además se generan `Grafico_Barras_9_Modelos.png` (gráfico de barras comparativo) y `Matrices_Confusion_9_Modelos.png` (matrices de confusión), ambos a 300 dpi.

## 11. Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `ERROR: Could not find a version that satisfies the requirement scikit-rebate` | El paquete se llama distinto en PyPI | `pip install skrebate` |
| `torch.cuda.is_available()` devuelve `False` o la versión termina en `+cpu` | Otro paquete reemplazó PyTorch | Repite el paso 8.5; si persiste, actualiza el driver NVIDIA |
| Aviso "Esta build de PyTorch NO incluye kernels para tu GPU" | PyTorch sin soporte `sm_120` | Instala PyTorch con `cu128` o superior |
| `TqdmWarning: IProgress not found` | Falta `ipywidgets` | `pip install ipywidgets` |
| `RuntimeError: No available kernel` (EXAONE / atención) | Kernels SDPA en RTX 50xx + Windows | El notebook ya fuerza `SDPBackend.MATH`; mantén `SDPA_SEGURO = True` |
| Cierre del kernel (`0xC0000005`) | Autocast `bf16` en modelos tabulares | Mantén `USE_AUTOCAST = False` |
| `WinError 1455` (archivo de paginación pequeño) | Cada worker de joblib carga `cublas64_12.dll` | El notebook usa `parallel_backend("threading")`; aumenta también el archivo de paginación de Windows |
| Error por `libiomp` / OpenMP duplicado | Conflicto entre torch, sklearn y autogluon | El notebook define `KMP_DUPLICATE_LIB_OK=TRUE` antes de importar torch |
| Bloqueos de Triton / inductor | `torch.compile` en Windows | El notebook define `TORCH_COMPILE_DISABLE=1` |
| `EnvironmentError: TabPFN necesita autenticación` | No hay token de TabPFN | Define `TABPFN_TOKEN` (sección 8.9) |
| `FileNotFoundError` al leer el CSV | Ruta del dataset distinta | Ajusta la ruta en la celda 3.1 |
| `CUDA out of memory` | VRAM insuficiente (7.9 GB usados en la tesis) | Cierra otros procesos y reinicia el kernel; ejecuta los modelos de uno en uno |

> Las variables de entorno (`TORCH_COMPILE_DISABLE`, `TOKENIZERS_PARALLELISM`, `KMP_DUPLICATE_LIB_OK`) deben definirse **antes** de importar `torch`, `tabicl` o `tabdpt`; por eso están al inicio del notebook.

## 12. Reproducibilidad y licencias

- **Semillas:** `random_state=42` en `RandomForestClassifier`, `RandomUnderSampler` y `StratifiedKFold`. `StratifiedGroupKFold` se usa sin barajado, por lo que sus particiones son deterministas.
- **Ruido:** el *Noise Engine* usa `np.random.normal` sin semilla fija, así que los resultados de la sección de robustez pueden variar ligeramente entre ejecuciones.
- **Validación:** todos los modelos de la comparativa final usan los mismos 5 folds agrupados por sujeto y la misma regla de métricas (F1 y AUC ponderados, AUC One-vs-Rest).
- **Licencias de modelos:** cada modelo fundacional tiene su propia licencia. En particular, los pesos de **TabFM** se distribuyen bajo una licencia **no comercial** (el código es Apache 2.0). Revisa las licencias de TabICL, TabDPT, TabPFN, Mitra/AutoGluon, iLTM y EXAONE-Tabular antes de cualquier uso comercial.
- **Credenciales:** el token de TabPFN es personal; no lo incluyas en el notebook ni en el repositorio.
- **Datos:** WESAD es un dataset de terceros; respeta sus condiciones de uso y cita la fuente original.

## Cita

```
Galarza, E. (2026). Predicción temprana de estrés usando datos fisiológicos multimodales. Tesis. [Universidad San Francisco de Quito / Maestria en ciencia de datos]
```
