import cv2
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# Load and resize grayscale image
img = cv2.imread('D:/Untitled-1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Sobel edge detection for generating labels
sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
edges = (np.abs(sobel) > 100).astype(np.uint8)

# Training data preparation
X = []
y = []
positions = []  # store (i, j) for plotting

for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):
        patch = img[i-1:i+2, j-1:j+2].flatten()
        X.append(patch)
        y.append(edges[i, j])
        positions.append((i, j))

X = np.array(X) / 255.0
y = np.array(y)

# Train perceptron
clf = Perceptron(max_iter=1000)
clf.fit(X, y)

# Predict edges using trained perceptron
y_pred = clf.predict(X)

# Draw green lines on a copy of original image
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for (i, j), pred in zip(positions, y_pred):
    if pred == 1:
        color_img[i, j] = [0, 255, 0]  # Green pixel

# Display the image with green edges
plt.figure(figsize=(6, 6))
plt.title("Edge Detected Image (Perceptron Output)")
plt.imshow(color_img)
plt.axis('off')
plt.show()

# Print learned weights and bias
print("Trained Perceptron Weights:", clf.coef_)
print("Trained Perceptron Bias:", clf.intercept_)
