from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from utils.cartoonify import cartoonify_image
from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
