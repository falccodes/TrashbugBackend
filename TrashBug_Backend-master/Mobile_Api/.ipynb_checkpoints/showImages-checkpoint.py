import os
from PIL import Image
import numpy as np
from flask import jsonify
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["GET"])
def showImage():
    folder='output'
    files=set()
    for filename in os.listdir(folder):
        if allowed_file(filename):
            files.add(filename)
    return render_template('showImages.html', filenames=files)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	#return redirect(url_for('', filename=filename), code=301)
    return send_from_directory("output", filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4884)