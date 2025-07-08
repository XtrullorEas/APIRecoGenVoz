import os
import librosa
import numpy as np

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
    # Script de prueba para procesar archivos de audio
    from utils import create_model
    import argparse
    
    parser = argparse.ArgumentParser(description="""Script de reconocimiento de género por archivo de audio""")
    parser.add_argument("-f", "--file", help="La ruta al archivo de audio", required=True)
    args = parser.parse_args()
    
    file = args.file
    if not os.path.isfile(file):
        print(f"❌ Error: El archivo {file} no existe")
        exit(1)
    
    try:
        # Construir el modelo
        model = create_model()
        # Cargar los pesos guardados/entrenados
        model.load_weights("results/model.h5")
        
        print(f"🎵 Procesando archivo: {file}")
        # Extraer características y redimensionar
        features = extract_feature(file, mel=True).reshape(1, -1)
        # ¡Predecir el género!
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "hombre" if male_prob > female_prob else "mujer"
        
        # ¡Mostrar el resultado!
        print(f"✅ Resultado: {gender}")
        print(f"📊 Probabilidades: Hombre: {male_prob*100:.2f}% | Mujer: {female_prob*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")
