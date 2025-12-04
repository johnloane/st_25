import socketio 
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10



