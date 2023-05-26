import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask
from flask import request, jsonify
import numpy as np
from lib import *
import dill
import time
import json
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

DATA_DIR = "/data/kai/SPCAM5/"
SCALER_DIR = "../../scalers/metrics/"


@app.post("/api/model")
def get_model():
    data = request.get_json()
    parameters = json.load(open("public/parameters.json"))

    return parameters[data["model"]]


@app.post("/api/data")
def get_image():
    # Return images for each level
    # Original, Model output (inverse transformed),
        # Original violin plot, Model output violin plot
        # and ordered lineplot 

    start = time.time()
    data = request.get_json()
    print(data)
    dataset = data["dataset"]
    date = data["date"]
    hour = int(data["hour"])
    level = int(data["level"])
    split = data["split"]
    scaler = data["scaler"]

    y_path = DATA_DIR + "inputs_%s.npy" % (date)
    y = np.load(y_path, mmap_mode="r").reshape(24, 96, 144, -1)[:, ::-1]

    og_y = y.copy()

    images = get_original_images(y, dataset, scaler, level, hour)


    print("Total time:", time.time() - start)
    return jsonify({"images": images})

# def get_loss_images(y, dataset, scaler, level, hour):




def get_original_images(y, dataset, scaler, level, hour):
    results = []

    if scaler == "min":
        y_scaler_max = np.load(SCALER_DIR + f"dataset_{dataset+1}_max.npy")
        y = y / y_scaler_max
        y[np.isnan(y)] = 0
    elif scaler == "minmax":
        y_scaler_minmax = dill.load(
            SCALER_DIR + f"dataset_{dataset+1}_y_scaler_minmax.pkl")
        y = y_scaler_minmax.transform(y)
    elif scaler == "standard":
        y_scaler_standard = dill.load(
            SCALER_DIR + f"dataset_{dataset+1}_y_scaler_standard.pkl")
        y = y_scaler_standard.transform(y)

    if level == 0:
        y = y[hour]
        results = []
        for l in range(26):
            plt.imshow(y[:, :, l], cmap="magma")
            plt.colorbar()
            plt.savefig("public/heatmap.jpg")
            plt.close()

            image = Image.open("public/heatmap.jpg")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            results.append(img_str)

        return results
    else:
        y = y[hour, :, :, level]

    plt.imshow(y, cmap="magma")
    plt.colorbar()
    plt.savefig("public/heatmap.jpg")
    plt.close()

    image = Image.open("public/heatmap.jpg")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return [img_str]

if __name__ == "__main__":

    # port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=5001)
    print('starting')
