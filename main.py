from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
# Corregir la importación - usar el nombre correcto de tu archivo
from test import record_to_file, extract_feature
from utils import create_model

app = Flask(__name__)
CORS(app)  # Agregar CORS para permitir peticiones desde otros dominios

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Cargar el modelo al iniciar
try:
    model = create_model()
    model.load_weights("results/model.h5")
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    return jsonify({
        'status': 'saludable',
        'model_loaded': model is not None,
        'message': 'API de Reconocimiento de Género'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
        
    # Verificar si se envió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo. Usa el endpoint /record para grabación con micrófono'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Tipo de archivo inválido. Permitidos: wav, mp3, flac, m4a'}), 400
    
    # Guardar archivo temporal
    filename = secure_filename(file.filename)
    temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
    
    try:
        file.save(temp_filename)
        
        # Procesar el archivo de audio
        features = extract_feature(temp_filename, mel=True).reshape(1, -1)
        male_prob = float(model.predict(features)[0][0])
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        
        # Determinar confianza
        confidence_score = max(male_prob, female_prob)
        if confidence_score > 0.8:
            confidence = "alto"
        elif confidence_score > 0.6:
            confidence = "medio"
        else:
            confidence = "bajo"
        
        return jsonify({
            'gender': gender,
            'male_probability': male_prob,
            'female_probability': female_prob,
            'confidence': confidence,
            'confidence_score': confidence_score
        })
        
    except Exception as e:
        return jsonify({'error': f'Error procesando audio: {str(e)}'}), 500
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.route('/record', methods=['POST'])
def record():
    """Endpoint específico para grabar desde el micrófono del servidor"""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
        
    try:
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.wav")
        
        # Nota: record_to_file requiere acceso al micrófono del servidor
        # Esto solo funcionará si el servidor tiene micrófono
        record_to_file(temp_filename)
        
        features = extract_feature(temp_filename, mel=True).reshape(1, -1)
        male_prob = float(model.predict(features)[0][0])
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        
        # Determinar confianza
        confidence_score = max(male_prob, female_prob)
        if confidence_score > 0.8:
            confidence = "alto"
        elif confidence_score > 0.6:
            confidence = "medio"
        else:
            confidence = "bajo"
        
        return jsonify({
            'gender': gender,
            'male_probability': male_prob,
            'female_probability': female_prob,
            'confidence': confidence,
            'confidence_score': confidence_score
        })
        
    except Exception as e:
        return jsonify({'error': f'Error grabando audio: {str(e)}'}), 500
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Archivo muy grande. Tamaño máximo es 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    # Crear directorio de uploads si no existe
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("🚀 Iniciando API de Reconocimiento de Género...")
    print("📁 Directorio de uploads creado/verificado")
    
    # Para desarrollo local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)