import pandas as pd
import numpy as np
import os
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split


label2int = {
    "male": 1,
    "female": 0
}


def load_data(vector_length=128):
    """Función para cargar el dataset de reconocimiento de género desde la carpeta `data`
    Después de la segunda ejecución, esto se cargará desde los archivos results/features.npy y results/labels.npy
    ¡ya que es mucho más rápido!"""
    # asegurar que la carpeta results existe
    if not os.path.isdir("results"):
        os.mkdir("results")
    # si las características y etiquetas ya están cargadas individualmente y agrupadas, cargarlas desde ahí
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y
    # leer dataframe
    df = pd.read_csv("balanced-all.csv")
    # obtener total de muestras
    n_samples = len(df)
    # obtener total de muestras masculinas
    n_male_samples = len(df[df['gender'] == 'male'])
    # obtener total de muestras femeninas
    n_female_samples = len(df[df['gender'] == 'female'])
    print("Total de muestras:", n_samples)
    print("Total de muestras masculinas:", n_male_samples)
    print("Total de muestras femeninas:", n_female_samples)
    # inicializar un array vacío para todas las características de audio
    X = np.zeros((n_samples, vector_length))
    # inicializar un array vacío para todas las etiquetas de audio (1 para masculino y 0 para femenino)
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Cargando datos", total=n_samples):
        features = np.load(filename)
        X[i] = features
        y[i] = label2int[gender]
    # guardar las características de audio y etiquetas en archivos
    # para no cargar cada una de ellas en la próxima ejecución
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y


def split_data(X, y, test_size=0.1, valid_size=0.1):
    # dividir conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    # dividir conjunto de entrenamiento y conjunto de validación
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    # devolver un diccionario de valores
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }


def create_model(vector_length=128):
    """5 capas densas ocultas de 256 unidades a 64, no es el mejor modelo, pero no está mal."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # una neurona de salida con función de activación sigmoide, 0 significa mujer, 1 significa hombre
    model.add(Dense(1, activation="sigmoid"))
    # usando binary crossentropy ya que es clasificación hombre/mujer (binaria)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # imprimir resumen del modelo
    model.summary()
    return model