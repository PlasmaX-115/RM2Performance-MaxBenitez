# Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
## Maximiliano Benítez Ahumada - A01752791

Para la realización de esta actividad se implementó el algoritmo KNN para clasificación.

# ¿Cómo funciona KNN?

K-Nearest Neighbors (k-NN): Algoritmo de aprendizaje supervisado para clasificación y regresión.

1. Entrenamiento: Almacena ejemplos con etiquetas conocidas.
2. Predicción (Clasificación): Encuentra los "k" ejemplos más cercanos en el conjunto de entrenamiento y asigna la etiqueta más común.
3. Predicción (Regresión): Encuentra los "k" ejemplos más cercanos y promedia sus valores para predecir un número.
4. El valor de "k" afecta la sensibilidad del algoritmo. Simple pero puede ser sensible al ruido y a la elección de la métrica de distancia.

# Manual de Instalación y Configuración de Miniconda y Entorno Virtual con scikit-learn para el uso del algoritmo KNN.

En este manua se presentan los pasos para instalar Miniconda, una versión minimalista de Anaconda, y cómo crear un entorno virtual utilizando Miniconda para trabajar con scikit-learn, una biblioteca de aprendizaje automático en Python con la cual se implementó KNN.

## Paso 1: Descargar Miniconda

1. Accede al sitio web de Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. Selecciona la versión de Miniconda adecuada para tu sistema operativo (Windows, macOS o Linux).

3. Descarga el instalador y sigue las instrucciones de instalación según tu sistema operativo.

## Paso 2: Instalar Miniconda

### Windows

1. Ejecuta el instalador descargado.

2. Acepta los términos y condiciones.

3. Elige la opción "Instalar solo para mí" y selecciona una ubicación para la instalación (por defecto, `C:\Users\<TuUsuario>\Miniconda3`).

4. Selecciona "Agregar Anaconda a mi PATH" durante la instalación.

5. Haz clic en "Instalar".

### macOS y Linux

1. Abre una terminal.

2. Navega hacia la ubicación del instalador descargado (usando el comando `cd`).

3. Ejecuta el comando de instalación: 

   ```
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```

   (El nombre del archivo puede variar según la versión descargada).

4. Acepta los términos y condiciones.

5. Sigue las instrucciones en pantalla y proporciona una ubicación para la instalación (por defecto, en tu directorio de inicio).

6. Acepta agregar el inicio de Miniconda al archivo `.bashrc` o `.zshrc` si se te pregunta.

## Paso 3: Crear un entorno virtual con scikit-learn

1. Abre una nueva terminal (si no tienes una abierta).

2. Crea un nuevo entorno virtual con el nombre "sklearn-env" y la versión de Python deseada (por ejemplo, Python 3.8):

   ```
   conda create -n sklearn-env python=3.8
   ```

3. Activa el entorno virtual:

   - En Windows:

     ```
     conda activate sklearn-env
     ```

   - En macOS y Linux:

     ```
     source activate sklearn-env
     ```

4. Instala scikit-learn en el entorno virtual:

   ```
   conda install scikit-learn
   ```

5. ¡Listo! Ahora tienes un entorno virtual con Miniconda y scikit-learn instalados.

## Paso 4: Usar el entorno virtual

1. Cada vez que quieras trabajar con scikit-learn, activa el entorno virtual:

   - En Windows:

     ```
     conda activate sklearn-env
     ```

   - En macOS y Linux:

     ```
     source activate sklearn-env
     ```
2. Posteriormente se podrá ejecutar el archivo 'train.py', el cual ejecuta todas las funciones necesarias de predicción.

## Output

El output de 'KNNEval.py' despliega las gráficas correspondientes a las matrices de confusión iterando el valor de K de 1 a 10.

Posteriormente despliega una gráfica que muestra la matriz de confusión con el valor de K más eficiente, es decir, aquel que tiene mayor precisión, así como una gráfica de los valores de la precisión del dataset de entrenamiento, pruebas y validación; valores con los cuales se determinan el sesgo y la varianza.


## Conclusións
1. Cuando haya terminado de trabajar, se debe desactivar el entorno virtual:

   ```
   conda deactivate
   ```


¡Se ha completado la creación de un entorno virtual, la instalación de scikit-learn y el testeo del algoritmo KNN! ¡Happy Hacking! 🚀



