from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":

    # port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='192.168.1.65', port=5000)