from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import time

app = Flask(__name__)
CORS(app)

model = tf.saved_model.load('magenta_model')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    print(img.shape)
    if max(img.shape) >= 1000:
        idx = np.argmax(img.shape)
        dim_ratio = img.shape[1-idx]/img.shape[idx]
        img = tf.image.resize(img,[1000,int(1000*dim_ratio)])
    print(img.shape)
    img = img[tf.newaxis,:]
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET','POST'])
def generate():
    contentfile = request.files['contentimage']
    stylefile = request.files['styleimage']
    print(contentfile.filename)
    if 'jpg' in contentfile.filename:
        content_path =  './images/content.jpg'
    elif 'jpeg' in contentfile.filename:
        content_path =  './images/content.jpeg'
    elif 'png' in contentfile.filename:
        content_path =  './images/content.png'
    if 'jpg' in stylefile.filename:
        style_path =  './images/style.jpg'
    elif 'jpeg' in stylefile.filename:
        style_path =  './images/style.jpeg'
    elif 'png' in stylefile.filename:
        style_path =  './images/content.png'
    
    print(content_path)
    print(style_path)
    contentfile.save(content_path)
    stylefile.save(style_path)
    content_img = load_image(content_path)
    print(content_img.shape)
    style_img = load_image(style_path)
    print(style_img.shape)
  
    stylized_image = model(tf.constant(content_img), tf.constant(style_img))[0]
    cv2.imwrite('./static/style_img.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

    return render_template('generate.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')