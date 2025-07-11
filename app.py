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
