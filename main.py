import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
# Load class names
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load the image
image_path = r'C:\object detection\img2.jpg'  # Update with your image path
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}. Please check the file path.")

# Ask the user for the object to detect
target_object = input("Enter the object to detect (e.g., car, person, etc.): ").lower()

# Ensure the object is in the class list
if target_object not in classes:
    print(f"'{target_object}' is not a valid object class. Check 'coco.names' for available classes.")
    exit()

# Image dimensions
height, width, _ = img.shape

# Preprocess the image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Detect objects
boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:  # Adjust confidence threshold as needed
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

# Initialize object count
object_count = 0

# Draw bounding boxes and labels on the image for the target object
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in indexes.flatten():
    if classes[class_ids[i]] == target_object:  # Filter for the target object
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Increment object count
        object_count += 1

# Display the total object count for the target object
cv2.putText(img, f"Detected '{target_object}': {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the final result
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the count in the terminal
print(f"Total '{target_object}' detected: {object_count}")
