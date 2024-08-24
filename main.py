import cv2
from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import base64

model = YOLO('yolov8l-face.pt')
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    print(request.files)
    if 'image' not in request.files:
        print('No image part in the request')
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        print('No selected file')
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Ler o arquivo em bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)

        # Converter bytes para uma imagem usando cv2
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print('Invalid image')
            return jsonify({"error": "Invalid image"}), 400

        imgBlur = blur_faces(img)

        _, img_encoded = cv2.imencode('.jpg', imgBlur)
        image_bytes = img_encoded.tobytes()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image


def blur_faces(img):
    # inference
    results = model.predict(img)[0]

    boxes = results.boxes

    for box in boxes:
        # blur image
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (81, 81), 90)

    return img


if __name__ == '__main__':
    app.run(debug=True, port=5520, host='0.0.0.0')
