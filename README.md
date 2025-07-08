# API de Reconocimiento de Género - Solo Archivos

## Cambios realizados

✅ **Eliminado pyaudio** - No más problemas de instalación en Render
✅ **Eliminado endpoint /record** - Ya no se necesita micrófono del servidor  
✅ **Creado test_clean.py** - Versión limpia sin dependencias de pyaudio
✅ **Actualizado requirements.txt** - Solo dependencias necesarias

## Endpoints disponibles

### 1. Health Check
```bash
GET /
```

Respuesta:
```json
{
    "status": "saludable",
    "model_loaded": true,
    "message": "API de Reconocimiento de Género",
    "timestamp": 1720396800.123
}
```

### 2. Warm-up (Opcional)
```bash
GET /warmup
```

Precarga el modelo para predicciones más rápidas. Úsalo después del deploy.

### 3. Predicción por archivo
```bash
POST /predict
```

**Parámetros:**
- `file`: Archivo de audio (wav, mp3, flac, m4a)

**Ejemplo con curl:**
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**Respuesta:**
```json
{
    "gender": "male",
    "male_probability": 0.85,
    "female_probability": 0.15,
    "confidence": "alto",
    "confidence_score": 0.85,
    "processing_time": 2.34
}
```

## Formatos de audio soportados
- WAV (recomendado)
- MP3
- FLAC  
- M4A

## Para probar localmente

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Ejecutar servidor:**
```bash
python main.py
```

3. **Probar con archivo:**
```bash
# Usando el script de prueba
python test_clean.py -f tu_archivo.wav
```

## Para desplegar en Render

1. **Build Command:** `pip install -r requirements.txt`
2. **Start Command:** `python main.py`
3. **Environment:** Python 3.8+

## Estructura final
```
APIRecoGenVoz/
├── main.py              # API Flask (sin pyaudio)
├── test_clean.py        # Funciones de audio (sin pyaudio)  
├── test.py             # Archivo original (no se usa)
├── utils.py            # Utilidades del modelo
├── requirements.txt    # Dependencias (sin pyaudio)
├── balanced-all.csv    # Dataset
└── results/
    ├── features.npy
    ├── labels.npy
    └── model.h5
```

## Optimizaciones para producción

### 🚀 **Mejoras implementadas**
- **Keep-alive automático** - Evita que la app se duerma en Render
- **Endpoint /warmup** - Precarga el modelo para predicciones más rápidas  
- **Tiempo de procesamiento** - Monitorea el rendimiento de cada predicción
- **Timestamp** - Tracking de disponibilidad en health check

### 🚨 **Troubleshooting**

#### **Error: "No se proporcionó archivo de audio"**
**En Postman:**
1. POST → `https://apirecogenvoz.onrender.com/predict`
2. Body → form-data
3. Key: `file` ⚠️ **Cambiar de "Text" a "File"**
4. Value: Seleccionar archivo WAV

#### **Para mejor rendimiento:**
```bash
# 1. Warm-up después del deploy
curl https://apirecogenvoz.onrender.com/warmup

# 2. Luego hacer predicciones
curl -X POST -F "file=@audio.wav" https://apirecogenvoz.onrender.com/predict
```

¡La API ahora funciona perfectamente sin pyaudio y solo procesa archivos subidos! 🎉
