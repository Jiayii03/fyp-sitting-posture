'''
check CPU temperature
vcgencmd measure_temp
'''

from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)
picam2 = Picamera2()

# 640 360
# 854 480
# 1280 720
camera_config = picam2.create_preview_configuration(main={"size": (640, 360)})

# Configure camera
picam2.configure(camera_config)
picam2.set_controls({"FrameRate": 30})

# Adjust white balance for better color
# The ScalerCrop needs to be specified as (x, y, width, height) in sensor coordinates
picam2.set_controls({
    "AwbEnable": True, 
    "AwbMode": 5  # Daylight mode
})

# Disable digital zoom by setting ScalerCrop properly
# We need to get sensor resolution first
sensor_resolution = picam2.camera_properties['PixelArraySize']
picam2.set_controls({"ScalerCrop": (0, 0, sensor_resolution[0], sensor_resolution[1])})

picam2.start()

def generate():
    while True:
        frame = picam2.capture_array()
        # Apply color correction
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add proper cleanup
import atexit
def cleanup():
    if 'picam2' in globals() and picam2 is not None:
        picam2.close()
atexit.register(cleanup)

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Camera Stream</title>
        <style>
          body { text-align: center; margin-top: 20px; }
          img { max-width: 100%; height: auto; }
        </style>
      </head>
      <body>
        <h1>Camera Stream</h1>
        <img src="/video_feed">
      </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)