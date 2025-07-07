import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Devuelve 'True' si está por debajo del umbral de 'silencio'"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Normaliza el volumen"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Recorta los espacios en blanco al inicio y al final"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Recortar por la izquierda
    snd_data = _trim(snd_data)

    # Recortar por la derecha
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Añade silencio al inicio y final de 'snd_data' de duración 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Graba una palabra o palabras desde el micrófono y 
    devuelve los datos como un array de shorts con signo.
    Normaliza el audio, recorta el silencio del 
    inicio y final, y añade 0.5 segundos de 
    sonido en blanco para asegurar que VLC y otros reproductores
    puedan reproducirlo sin cortarse.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Graba desde el micrófono y guarda los datos resultantes en 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



def extract_feature(file_name, **kwargs):
    """
    Extrae características del archivo de audio `file_name`
        Características soportadas:
            - MFCC (mfcc)
            - Chroma (chroma)
            - Frecuencia MEL del Espectrograma (mel)
            - Contraste (contrast)
            - Tonnetz (tonnetz)
        Ejemplo:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


if __name__ == "__main__":
    # cargar el modelo guardado (después del entrenamiento)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    from utils import load_data, split_data, create_model
    import argparse
    parser = argparse.ArgumentParser(description="""Script de reconocimiento de género, esto cargará el modelo que entrenaste, 
                                    y realizará inferencia en una muestra que proporciones (usando tu voz o un archivo)""")
    parser.add_argument("-f", "--file", help="La ruta al archivo, preferiblemente en formato WAV")
    args = parser.parse_args()
    file = args.file
    # construir el modelo
    model = create_model()
    # cargar los pesos guardados/entrenados
    model.load_weights("results/model.h5")
    if not file or not os.path.isfile(file):
        # si no se proporciona archivo, o no existe, usa tu voz
        print("Por favor habla")
        # pon el nombre del archivo aquí
        file = "test.wav"
        # grabar el archivo (comienza a hablar)
        record_to_file(file)
    # extraer características y redimensionar
    features = extract_feature(file, mel=True).reshape(1, -1)
    # ¡predecir el género!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "hombre" if male_prob > female_prob else "mujer"
    # ¡mostrar el resultado!
    print("Resultado:", gender)
    print(f"Probabilidades:     Hombre: {male_prob*100:.2f}%    Mujer: {female_prob*100:.2f}%")
