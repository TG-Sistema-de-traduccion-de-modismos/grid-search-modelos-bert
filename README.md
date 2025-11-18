# Grid Search para Modelos BERT en Español

Sistema automatizado de búsqueda de hiperparámetros para modelos BERT en español utilizando Docker y GPU.

## Descripción

Este proyecto realiza un grid search exhaustivo sobre tres modelos BERT preentrenados en español para tareas de clasificación de secuencias. El sistema evalúa múltiples combinaciones de hiperparámetros y genera reportes detallados de rendimiento.

## Modelos Utilizados

| Modelo | HuggingFace Link | Descripción |
|--------|------------------|-------------|
| **BETO** | [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) | BERT base para español con Whole Word Masking |
| **BERTIN** | [bertin-project/bertin-roberta-base-spanish](https://huggingface.co/bertin-project/bertin-roberta-base-spanish) | RoBERTa base entrenado en español |
| **ALBERTO** | [CenIA/albert-base-spanish](https://huggingface.co/CenIA/albert-base-spanish) | ALBERT base para español |

## Características

- **Grid Search Automatizado**: Evalúa 67 combinaciones de hiperparámetros por modelo (201 experimentos totales)
- **Balanceo de Clases**: Implementa pesos de clase automáticos para datos desbalanceados
- **Marcado de Contexto**: Sistema de marcado `[TGT]` para destacar palabras objetivo en contexto
- **Métricas Completas**: Accuracy, Precision, Recall y F1-Score
- **Guardado Incremental**: Resultados parciales cada 10 experimentos
- **Optimizado para GPU**: Soporte CUDA 12.8 con cuDNN

## Requisitos

- Docker con soporte para GPU (NVIDIA Docker)
- NVIDIA Driver compatible con CUDA 12.8
- GPU con al menos 8GB VRAM (recomendado)
- Dataset en formato JSON (`dataset.json`)

## Instalación y Uso

### 1. Preparar el Dataset

Crea un archivo `dataset.json` con el siguiente formato:

```json
[
  {
    "raiz": "palabraraiz",
    "contextos": [
      "Contexto de ejemplo con palabraraiz",
      "Otro contexto diferente"
    ],
    "etiquetas": [
      "etiqueta1",
      "etiqueta2"
    ]
  }
]
```

### 2. Construir la Imagen Docker

```bash
docker build -t grid-search-bert .
```

### 3. Ejecutar el Grid Search

```bash
docker run --gpus all -v ${PWD}:/app/output grid-search-bert
```

**En Windows PowerShell:**
```powershell
docker run --gpus all -v ${PWD}:/app/output grid-search-bert
```

**En Windows CMD:**
```cmd
docker run --gpus all -v %cd%:/app/output grid-search-bert
```

## Hiperparámetros Explorados

| Parámetro | Valores |
|-----------|---------|
| `learning_rate` | 1e-5, 2e-5, 3e-5, 5e-5 |
| `num_train_epochs` | 3, 5, 8, 10 |
| `per_device_train_batch_size` | 8, 16 |
| `weight_decay` | 0.0, 0.01, 0.05 |
| `warmup_steps` | 0, 50, 100 |
| `lr_scheduler_type` | linear, cosine |

## Resultados

El sistema genera tres archivos Excel:

1. **`resultados_grid_search.xlsx`**: Todos los experimentos con métricas completas
2. **`mejores_modelos.xlsx`**: Mejor configuración por modelo
3. **`resultados_parciales.xlsx`**: Guardado incremental durante la ejecución

### Estructura de Resultados

Cada experimento incluye:
- Identificador único (`run_id`)
- Modelo evaluado
- Timestamp de ejecución
- Todos los hiperparámetros probados
- Métricas de entrenamiento y evaluación
- Estado de finalización (completado/error)

## Características Técnicas

### Preprocesamiento
- Marcado automático de palabras objetivo con tags `[TGT]`
- Tokenización adaptativa (máximo 128 tokens)
- División estratificada 80/20 (train/test)

### Entrenamiento
- Función de pérdida con pesos de clase balanceados
- Evaluación por época
- Semilla fija (42) para reproducibilidad
- Limpieza automática de caché GPU entre experimentos

### Optimizaciones Docker
- Imagen base CUDA 12.8 con cuDNN
- Entorno virtual Python aislado
- Cache de pip optimizado
- Instalación de PyTorch desde índice CUDA específico

##  Archivos del Proyecto

```
.
├── Dockerfile              # Configuración del contenedor
├── requirements.txt        # Dependencias Python
├── grid_search.py         # Script principal
├── dataset.json           # Dataset de entrada
└── README.md              # Esta documentación
```

## Troubleshooting

### Error: "CUDA out of memory"
- Reduce `per_device_train_batch_size` en `GRID_PARAMS`
- Reduce el número de combinaciones (`max_per_model`)

### Error: "No GPU available"
- Verifica la instalación de NVIDIA Docker: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
- Asegúrate de usar `--gpus all` en el comando docker run

### Dataset no encontrado
- Verifica que `dataset.json` esté en el directorio actual
- Comprueba el formato JSON del dataset

## Notas

- El proceso completo puede tomar **varias horas** dependiendo del tamaño del dataset y hardware
- Se recomienda al menos **16GB de RAM** del sistema
- Los checkpoints NO se guardan para ahorrar espacio (`save_strategy="no"`)
- Progreso se muestra cada 10 experimentos completados



