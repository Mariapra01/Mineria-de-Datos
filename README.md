# Minería de Datos Clínicos: 
## Resumen

Este proyecto de Minería de Datos Clínicos tiene como objetivo explorar, modelar y mejorar la representación de un conjunto de datos sobre salud mental mediante técnicas de aprendizaje automático y generación de datos sintéticos. Tras un exhaustivo análisis exploratorio (EDA) y preprocesamiento del dataset original, se ha desarrollado un modelo de Autoencoder Variacional (VAE) capaz de generar perfiles sintéticos clínicamente plausibles.

El modelo se ha entrenado durante 30 épocas sobre datos preprocesados y escalados, utilizando la función de pérdida combinada (reconstrucción + divergencia KL). Se han generado nuevos datos sintéticos desde el espacio latente, y posteriormente se ha evaluado su utilidad entrenando modelos de clasificación clásicos.

Los resultados demuestran que el uso de datos sintéticos no solo permite preservar la estructura estadística de los datos originales, sino que mejora significativamente el rendimiento de modelos supervisados como Gradient Boosting o Random Forest en términos de precisión y generalización. Además, se ha logrado una mejor detección de la clase minoritaria (pacientes que reciben tratamiento), lo cual tiene un gran valor en aplicaciones clínicas reales.

Este enfoque demuestra la aplicabilidad de los modelos generativos como apoyo a tareas de clasificación en contextos de datos clínicos, aportando robustez y capacidad de generalización a partir de un dataset limitado.

## Generación de Datos Sintéticos con VAE

Este proyecto aplica técnicas de minería de datos y aprendizaje profundo sobre un dataset de salud mental. Se ha llevado a cabo un análisis exploratorio, preprocesamiento y entrenamiento de un Autoencoder Variacional (VAE) para la generación de datos sintéticos que simulan perfiles clínicos realistas.

## Estado del Arte
Los Autoencoders Variacionales (VAE) representan una evolución de los autoencoders tradicionales, incorporando fundamentos probabilísticos para aprender representaciones latentes continuas y generativas. A diferencia de los autoencoders clásicos, que codifican una entrada en un punto fijo del espacio latente, los VAE modelan una distribución latente (usualmente gaussiana), lo que permite la generación de datos sintéticos plausibles mediante muestreo desde ese espacio.

En este proyecto, el VAE se ha diseñado con una arquitectura densa simple, adaptada a datos clínicos estructurados (no imagen ni texto), e incluye:

Encoder que transforma los datos originales en dos vectores: z_mean y z_log_var, que definen la media y varianza logarítmica de una distribución normal multivariante.

Sampling con reparametrización (truco de Kingma & Welling, 2013), para permitir el flujo de gradientes durante el entrenamiento.

Decoder que intenta reconstruir los datos originales a partir de muestras z tomadas de la distribución latente.

El entrenamiento minimiza una función de pérdida compuesta por:

Pérdida de reconstrucción (MSE): mide la diferencia entre entrada y salida.

Divergencia KL (Kullback-Leibler): penaliza desviaciones respecto a la distribución normal estándar, regulando el espacio latente.

Los VAE han demostrado ser eficaces en tareas de generación de datos, reducción de dimensionalidad y aprendizaje no supervisado. En este trabajo, se evalúa su utilidad para aumentar datos clínicos y mejorar el rendimiento de modelos supervisados clásicos, lo que representa una contribución original en un contexto académico de salud mental.

## Contenidos del repositorio

- `MentalHealth.ipynb`: Análisis exploratorio de datos (EDA) y visualización.
- `VAE_v1.ipynb`: Construcción, entrenamiento y evaluación del modelo VAE.
- `informe_eda_salud_mental.html`: Informe interactivo del análisis exploratorio (YData Profiling).
- `vae_diagrama.jpg` / `vae_sampling.jpg`: Ilustraciones del flujo del VAE y proceso de muestreo.

## Objetivos

- Comprender el comportamiento de los datos clínicos mediante técnicas EDA.
- Generar datos sintéticos realistas utilizando un Autoencoder Variacional.
- Evaluar si los datos generados pueden mejorar el rendimiento de modelos de clasificación.
- Comparar modelos clásicos (Logistic Regression, Random Forest, Naive Bayes, KNN, Gradient Boosting) con datos reales vs. sintéticos.

## Metodología

En el notebook viene toda la implementación del VAE, así como las métricas usadas para medir el nivel de los datos generados, y los resultados obtenidos en los entrenamientos de los modelos de ML (con datos originales y datos sintéticos)

## Notas

- Los datos se escalaron a [0,1] para ajustarse a la salida sigmoide del VAE.
- Se implementó sampling con reparametrización para permitir la generación de múltiples perfiles plausibles a partir de la distribución latente de cada paciente.

## Resultados destacados

| Modelo              | Accuracy (Datos Originales) | Accuracy (Datos Sintéticos) |
|---------------------|-----------------------------|------------------------------|
| Logistic Regression | 0.6371                      | 0.7209                       |
| Random Forest       | 0.5488                      | 0.7587                       |
| Naive Bayes         | 0.6325                      | 0.7029                       |
| KNN                 | 0.5345                      | 0.6519                       |
| Gradient Boosting   | 0.6862                      | 0.7650                       |

Los datos sintéticos generados por el VAE mostraron una mejora significativa en la precisión de los modelos de clasificación.

## Requisitos

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

## Visualizaciones

Se incluyen gráficos de:

- Distribución de atributos originales vs. sintéticos.
- Evolución de las pérdidas del VAE.
- Representación visual del espacio latente.


## Licencia

Este proyecto es parte de una prueba académica y se publica únicamente con fines educativos.
