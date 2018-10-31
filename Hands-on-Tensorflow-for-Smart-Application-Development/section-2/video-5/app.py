from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import io
import os

import random
import string
import datetime

import base64
from PIL import Image, ImageFile

from flask import request, jsonify, Flask

import numpy as np
import tensorflow as tf

import label_image

model_file = "tf_files/retrained_graph.pb"
label_file = "tf_files/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"

g = tf.Graph()
with g.as_default():
    graph = label_image.load_graph(model_file)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])

def predict():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded))

    destination = "images"
    if not os.path.exists(destination):
        os.makedirs(destination)

    now = datetime.datetime.now()
    rand_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    file_name = os.path.join(destination, str(now.strftime("%Y-%m-%d-%H-%M-%S-"))+rand_str+'.jpg')

    try:
        img.save(os.path.join(file_name), "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        img.save(file_name, "JPEG", quality=80, optimize=True, progressive=True)

    t = label_image.read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
        end=time.time()

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = label_image.load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    for i in top_k:
        print(template.format(labels[i], results[i]))

    response = {
        'prediction': {
            'prediction': labels[0],
            'value' : str(results[0])
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
