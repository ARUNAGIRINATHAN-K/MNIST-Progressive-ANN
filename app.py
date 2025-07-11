from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from utils.cartoonify import cartoonify_image
from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Cartoonify
        output_image = cartoonify_image(input_path)
        output_filename = f"cartoon_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        output_image.save(output_path)

        return render_template('result.html', original=input_path, cartoon=output_path)

    return redirect(url_for('index'))

@app.route('/static/<path:path>')
def send_file(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
