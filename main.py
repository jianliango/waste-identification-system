import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model('waste_identification_model.h5')

# Define Class Indices
class_indices = {'Metal': 0, 'Plastic': 1, 'Wood': 2}
inv_class_indices = {v: k for k, v in class_indices.items()}

# Prediction Function 
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class_index = np.argmax(pred)
    pred_label = inv_class_indices[pred_class_index]
    confidence = float(pred[0][pred_class_index])

    return pred_label, confidence

# GUI Window
root = tk.Tk()
root.title("Waste Identification System")
root.geometry("600x600")

# UI Elements
image_label = Label(root, text="No image loaded", font=("Arial", 14))
image_label.pack(pady=10)

prediction_label = Label(root, text="", font=("Arial", 18, "bold"))
prediction_label.pack(pady=10)

# Load Image and Get Predict Result
def load_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    # Display the image on UI Element
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Run prediction
    label, confidence = predict_image(file_path)

    prediction_label.config(
        text=f"Prediction: {label}\nConfidence: {confidence*100:.2f}%"
    )

# Open File Button 
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

btn1 = Button(btn_frame, text="Load Image Files", width=15, command=load_and_predict)
btn1.grid(row=0, column=0, padx=10)

# Start GUI
root.mainloop()
