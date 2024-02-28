import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from tensorflow.keras.models import load_model
import pandas as pd
import datetime

# Load the license plate detection cascade classifier
plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

# Load the character recognition model
model = load_model('character_recognition_model_rnn.h5')

# Function to detect license plate from an image
def detect_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :]
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2, y), (x+w-3, y+h-5), (51, 181, 155), 3)
    return plate_img, plate

# Function to segment characters from the license plate
def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    char_list = find_contours(dimensions, img_binary_lp)
    return char_list

# Function to preprocess and classify characters
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for ch in char:
        img_resized = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img_fixed = fix_dimension(img_resized)
        img_normalized = img_fixed / 255.0
        img_reshaped = img_normalized.reshape(1, 28, 28, 1)
        predictions = model.predict(img_reshaped)[0]
        predicted_class = np.argmax(predictions)
        character = dic[predicted_class]
        output.append(character)

    plate_number = ''.join(output)
    return plate_number

# Function to find contours in an image
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)
            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0
            img_res.append(char_copy)
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)
    return img_res

# Function to preprocess images for classification
def fix_dimension(img):
    return img[:, :, np.newaxis]

# Function to load and preprocess an image
def load_and_preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_bytes = uploaded_file.getvalue()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function to display images using Streamlit
def display_image(image, title=''):
    st.image(image, caption=title, use_column_width=True)

# Function to check if a record is found in the dataset
def check_record_found(df, plate_number):
    return plate_number in df['Vehicle Registration No.'].values

# Function to calculate the time difference between current date and installment date
def calculate_time_difference(installment_date):
    installment_date = installment_date.iloc[0]  # Select the first element
    current_date = datetime.datetime.now()
    time_difference = current_date - installment_date
    return time_difference.days


# Main function to run the app
def main():
    st.title("License Plate Recognition App")
    st.write("Upload an image containing a license plate to get started.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform license plate detection
        img = load_and_preprocess_image(uploaded_file)
        plate_img, plate = detect_plate(img)
        display_image(plate_img, title='Detected License Plate')

        # Perform character segmentation and recognition
        char = segment_characters(plate)
        plate_number = show_results(char)

        # Display segmented characters and their predicted values
        st.write("Predicted License Plate Number:")
        st.write(plate_number)
        # st.write("Segmented Characters:")
        # for i, ch in enumerate(char):
        #     img = cv2.resize(ch, (4, 4), interpolation=cv2.INTER_AREA)
        #     display_image(img, title=f"Predicted: {plate_number[i]}")

        # Load dataset
        df = pd.read_excel('CNG-Dataset.xlsx')

        # Check if the license plate number exists in the dataset
        if check_record_found(df, plate_number):
            result = df[df['Vehicle Registration No.'] == plate_number]
            st.write("Record Found:")
            st.write(result[['Installment Date', 'Customer Name', 'Vehicle Registration No.']])

            installment_date = pd.to_datetime(result['Installment Date'])
            time_difference = calculate_time_difference(installment_date)

            no_filling_threshold = 3 * 365  # 3 years

            if time_difference >= no_filling_threshold:
                st.error("🙅🚫 No CNG filling allowed, more than 3 years have passed since CNG tank fitting.")
            elif time_difference == 1090:
                st.warning("Only 5 days remaining.")
            elif time_difference == 1091:
                st.warning("Only 4 days remaining.")
            elif time_difference == 1092:
                st.warning("Only 3 days remaining.")
            elif time_difference == 1093:
                st.warning("Only 2 days remaining.")
            elif time_difference == 1094:
                st.warning("Only 1 day remaining.")
            else:
                st.success("CNG Filling Allowed")
        else:
            st.warning("Record not found")

    # Display developer info
    st.header("About the Developer")
    st.image("developer_photo.jpg", width=180)
    st.write("This app was developed by :red[Rohit Vikas Dhotre].")
    st.write("You can contact me at :red[rohitdhotre29112001@gmail.com].")

if __name__ == '__main__':
    main()
