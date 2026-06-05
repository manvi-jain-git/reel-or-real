from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import shutil
from analyzer.scorer import analyze_video

app = Flask(__name__)
CORS(app)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    
    # Ensure temp dir exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    try:
        result = analyze_video(url, TEMP_DIR)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    app.run(debug=True, port=5000)
