import numpy as np
from keras_preprocessing import image
import cv2
import os
from tensorflow.keras.models import load_model

# Load model
try:
    model = load_model('model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Start video capture
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
classes = ['Chuối', 'Dâu tây', 'Dứa', 'Khế', 'Măng cụt', 'Táo']
new_model = load_model('model.keras')

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to capture image")
        break

    # Save the current frame as an image
    file_path = f'C:\\csudai\\final_{i}.jpg'
    result = cv2.imwrite(file_path, frame)
    if result:
        print(f"Image saved successfully to {file_path}")
    else:
        print(f"Failed to save image to {file_path}")
        continue

    try:
        # Load image for prediction
        test_image = image.load_img(file_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict class dự đoán
        result = model.predict(test_image)
        predicted_class = np.argmax(result[0])
        prediction = classes[predicted_class]

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 50)  # Vị trí text trên khung hình
        font_scale = 1
        font_color = (0, 255, 0)# Màu xanh lá cây
        thickness = 2

        # Hiển thị nhãn lên khung hình
        cv2.putText(frame, prediction, position, font, font_scale, font_color, thickness)

    except FileNotFoundError:
        print(f"Image file {file_path} not found")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Delete the saved image
    if os.path.exists(file_path):
        os.remove(file_path)

    # Show the video frame with the prediction label
    cv2.imshow('Camera', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1

# Release video capture and close windows
vid.release()
cv2.destroyAllWindows()
