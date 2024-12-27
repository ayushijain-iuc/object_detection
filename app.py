import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Object Detection Function
def detect_objects(image_path, target_object):
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, "Invalid image file."

    # Image dimensions
    height, width, _ = img.shape

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Variables for detection
    boxes = []
    confidences = []
    class_ids = []
    object_count = 0

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes for the target object
    for i in indexes.flatten():
        if classes[class_ids[i]] == target_object.lower():
            x, y, w, h = boxes[i]
            color = colors[class_ids[i]]
            label = f"{classes[class_ids[i]]} {round(confidences[i], 2)}"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            object_count += 1

    # Save the processed image
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path, object_count, None

# Route: Render HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle file upload and detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files or 'object' not in request.form:
        return "No file or object specified.", 400

    file = request.files['image']
    target_object = request.form['object']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Perform object detection
        processed_path, object_count, error = detect_objects(image_path, target_object)
        if error:
            return error, 400

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Object Detection Results</title>
        <style>
            body {{ 
                text-align: center; 
                font-family: Arial, sans-serif; 
                background-color: #f4f4f4; 
                background-image: url('{url_for('static', filename='image/images (1).jpg')}');
                background-size: cover; /* Ensure the image covers the entire screen */
                background-position: center; /* Center the image */
                margin: 0;
                padding: 0;
            }}
            .container {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 0 10px gray; 
                max-width: 600px; 
                margin: 50px auto; 
            }}
            img {{ 
                margin-top: 20px; 
                width: 100%; 
                max-height: 400px; 
                object-fit: contain; 
            }}
            .highlight {{ 
                font-size: 1.5em; 
                font-weight: bold; 
                color: #ff0000; /* Highlighted Red Text */
                background-color: #ffff99; /* Yellow Background */
                padding: 5px 10px; 
                border-radius: 5px; 
                display: inline-block; 
                margin: 10px 0; 
            }}
            .button {{ 
                display: inline-block; 
                margin-top: 20px; 
                padding: 10px 20px; 
                background-color: #28a745; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                font-size: 16px; 
            }}
            .button:hover {{ 
                background-color: #218838; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Object Detection Results</h1>
            <p class="highlight">Total '<strong>{target_object}</strong>' detected: <strong>{object_count}</strong></p>
            <img src="/processed/{filename}" alt="Processed Image">
            <a href="/" class="button">⬅ Go Back</a>
        </div>
    </body>
    </html>
    """



# Route: Serve processed images
@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
