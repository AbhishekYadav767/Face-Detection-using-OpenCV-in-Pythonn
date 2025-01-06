# Import necessary packages
import numpy as np
import cv2

# Define file paths for the image, prototxt, and model
image_path = "rooster.jpg"  # Replace with the path to your input image
prototxt_path = "deploy.prototxt.txt"  # Replace with the path to your prototxt file
model_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Replace with the path to your Caffe model
confidence_threshold = 0.5  # Minimum confidence to filter weak detections

# Load the serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the input image and get its dimensions
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Construct an input blob for the image by resizing to 300x300 pixels and normalizing
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network to obtain the detections
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence (probability) of the detection
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > confidence_threshold:
        # Compute the (x, y)-coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box and confidence on the image
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
