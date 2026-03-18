from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def home():
    return "Backend is running 🚀"
def send_update(frame, data):
    socketio.emit("report", data)



if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)