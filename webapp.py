
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
import numpy as np

# # Load the trained model
model = load_model('my_model.hdf5')

# # Define the labels for the classes
labels = ['complex',	'frog_eye_leaf_spot',	'healthy',	'powdery_mildew',	'rust',	'scab']

# Create a function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((384,384))
    
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)

    img_array = np.array(img)
    img_array = img_array / 255.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.set_page_config(page_title="Apple leaf disease classification", page_icon=":guardsman:", layout="wide")
    st.title("Apple leaf disease classification")

    # Get the image from the user
    image_file = st.file_uploader("Upload an image of apple leaf", type=["jpg", "png",'jpeg'])
    if image_file is not None:
        image_ = Image.open(image_file)
        st.image(image_, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image_file)

        # Make a prediction
        predictions = model.predict(img_array)

        # Get the class label and class probability
        label_index = np.argmax(predictions)
        # class_probability = np.max(predictions)
        class_label = labels[label_index]

        # Show the results
        st.success("The model predicted that this image is a: " + class_label)
        # st.write("With probability: " + str(class_probability))

if __name__=='__main__':
    main()
