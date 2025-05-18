# 🧠 Minería de Datos Clínicos: Generación de Datos Sintéticos con VAE

Este proyecto aplica técnicas de minería de datos y aprendizaje profundo sobre un dataset de salud mental. Se ha llevado a cabo un análisis exploratorio, preprocesamiento y entrenamiento de un Autoencoder Variacional (VAE) para la generación de datos sintéticos que simulan perfiles clínicos realistas.

## 📂 Contenidos del repositorio

- `MentalHealth.ipynb`: Análisis exploratorio de datos (EDA) y visualización.
- `VAE_v1.ipynb`: Construcción, entrenamiento y evaluación del modelo VAE.
- `informe_eda_salud_mental.html`: Informe interactivo del análisis exploratorio (YData Profiling).
- `vae_diagrama.jpg` / `vae_sampling.jpg`: Ilustraciones del flujo del VAE y proceso de muestreo.

## 🧪 Objetivos

- Comprender el comportamiento de los datos clínicos mediante técnicas EDA.
- Generar datos sintéticos realistas utilizando un Autoencoder Variacional.
- Evaluar si los datos generados pueden mejorar el rendimiento de modelos de clasificación.
- Comparar modelos clásicos (Logistic Regression, Random Forest, Naive Bayes, KNN, Gradient Boosting) con datos reales vs. sintéticos.

## 📊 Resultados destacados

| Modelo              | Accuracy (Datos Originales) | Accuracy (Datos Sintéticos) |
|---------------------|-----------------------------|------------------------------|
| Logistic Regression | 0.6371                      | 0.7209                       |
| Random Forest       | 0.5488                      | 0.7587                       |
| Naive Bayes         | 0.6325                      | 0.7029                       |
| KNN                 | 0.5345                      | 0.6519                       |
| Gradient Boosting   | 0.6862                      | 0.7650                       |

Los datos sintéticos generados por el VAE mostraron una mejora significativa en la precisión de los modelos de clasificación.

## 🛠️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

Algunas librerías utilizadas:

- `tensorflow`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `ydata-profiling`
- `pandas`, `numpy`

## 🖼️ Visualizaciones

Se incluyen gráficos de:

- Distribución de atributos originales vs. sintéticos.
- Evolución de las pérdidas del VAE.
- Representación visual del espacio latente.

## 📌 Notas

- Los datos se escalaron a [0,1] para ajustarse a la salida sigmoide del VAE.
- Se implementó sampling con reparametrización para permitir la generación de múltiples perfiles plausibles a partir de la distribución latente de cada paciente.

## 📜 Licencia

Este proyecto es parte de una prueba académica y se publica únicamente con fines educativos.
