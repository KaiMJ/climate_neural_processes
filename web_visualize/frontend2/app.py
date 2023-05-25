from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.post("/api/data")
def get_image(timestep):
    data = request.get_json()
    print(data)
    return "image"


if __name__ == "__main__":

    # port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='192.168.1.65', port=5000)