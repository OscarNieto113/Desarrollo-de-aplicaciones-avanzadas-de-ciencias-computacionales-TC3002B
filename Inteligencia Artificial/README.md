# Descripción del Proyecto

## Sobre el Conjunto de Datos
El conjunto de datos original de Kaggle contiene todas las imágenes sin una estructura definida por clase. Con el fin de mejorar la eficiencia del desarrollo del modelo y reducir posibles desequilibrios, las imágenes se dividirán en tres categorías principales:

- **Entrenamiento (Train):** Contiene el 70% de las imágenes de cada clase y se utiliza para entrenar el modelo.
- **Validación (Validation):** Aquí se evalúa el rendimiento del modelo durante el entrenamiento, permitiendo ajustar los hiperparámetros antes de la fase de prueba. Contendrá el 15% de las imágenes de cada clase.
- **Prueba (Test):** Permite evaluar el rendimiento del modelo frente a imágenes que nunca ha visto durante la fase de entrenamiento. También contendrá el 15% de las imágenes de cada clase.

## Descarga del Conjunto de Datos
Puedes descargar el conjunto de datos original de Kaggle desde [aquí](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier?select=Data). Ten en cuenta que las imágenes en este conjunto de datos no son propiedad del creador de este proyecto.

## Preprocesamiento de los Datos
El preprocesamiento de imágenes es una fase relevante para preparar los datos antes de entrenar un modelo de clasificación de imágenes. En este proyecto, se utiliza `ImageDataGenerator` de TensorFlow para realizar una serie de transformaciones en las imágenes del conjunto de datos.

### Detalles del Preprocesamiento
- **Redimensionamiento de Imágenes:** Todas las imágenes se redimensionan a 150x150 píxeles para asegurar una entrada consistente en el modelo, independientemente de las dimensiones originales de las imágenes.
- **Normalización de los Valores de los Píxeles:** Se escalan los valores de los píxeles de las imágenes al rango [0, 1].
- **Data Augmentation (Aumento de Datos):** Se aplican diversas transformaciones en el conjunto de entrenamiento para incrementar la diversidad del conjunto de datos, como rotaciones, desplazamientos, cortes, zoom y volteo horizontal.

### Generador de Datos
`ImageDataGenerator` genera lotes de datos de imágenes con un tamaño de 64 imágenes por lote. Para fines didácticos, se guarda ejemplos de imágenes aumentadas en la carpeta "augmented" con el prefijo "aug" y en formato PNG.

## Ejemplo de Imágenes Generadas
![Ejemplo de Imágenes Generadas](link_to_image)

En el archivo "model.py", se mantiene la misma arquitectura descrita anteriormente para el preprocesamiento de las imágenes de entrenamiento. Sin embargo, se ha agregado un preprocesamiento similar para los conjuntos de validación y prueba, conservando una arquitectura simplificada para estos conjuntos para garantizar una evaluación precisa del rendimiento del modelo en situaciones del mundo real.
