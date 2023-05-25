from flask import Flask
from flask import request, jsonify
import numpy as np
from lib import *
import dill
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64, io
from PIL import Image

app = Flask(__name__)

DATA_DIR = "../../data/SPCAM5/"
SCALER_DIR = "../../scalers/metrics/"


@app.post("/api/data")
def get_image():
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

        print("Total time:", time.time() - start)
        return jsonify({"images": results})
    else:
        y = y[hour, :, :, level]

    print(y.shape)

    plt.imshow(y, cmap="magma")
    plt.colorbar()
    plt.savefig("public/heatmap.jpg")
    plt.close()

    image = Image.open("public/heatmap.jpg")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    print("Total time:", time.time() - start)
    return jsonify({"image": img_str})


if __name__ == "__main__":

    # port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=5001)
    print('starting')
