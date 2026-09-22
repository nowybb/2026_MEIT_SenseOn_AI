from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()


def update_frame(frame):
    global latest_frame

    with frame_lock:
        latest_frame = frame.copy()


def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
        <head>
            <title>SenseOn Camera</title>
        </head>
        <body>
            <h2>SenseOn Live View</h2>
            <img src="/video" width="640">
        </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def run_stream_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )