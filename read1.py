import os.path
import io
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import random
import sklearn 
import pickle
import streamlit.components.v1 as components
import base64

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#st.write("ik")
#submit = st.button("Click here")
# from PIL import Image
# ////////////////////////////////////////////////
##st.title("Explicit Content Detection")

# submit = st.button("Home")
# submit = st.button("LogIn")
# submit = st.button("LogOut")


st.markdown("<h1 style='text-align: center; color: white; background: rgba(0, 65, 69, 0.86)'>Explicit Content Detection</h1>", unsafe_allow_html=True)
# Define CSS style for the background image
# Define CSS style for the background image




def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('12.png')


# def load_image(image_file):
#     img = Image.open(image_file)

#     return img



st.markdown("<h1 style='color: white; font-size: 24px;'>Upload Image</h1>", unsafe_allow_html=True)
image_file = st.file_uploader("", type=["png", "jpg", "jpeg", "webp"])
st.subheader("RESULTS")
#st.markdown("<h3 style='text-align: center; color: red;'>Some title</h3>", unsafe_allow_html=True)

# st.markdown("<h1 style='color: white;font-size: 24px;'>Enter link</h1>", unsafe_allow_html=True)
# link_t = st.text_area("", height=100)



# st.markdown("<h1 style='color: white;font-size: 24px;'>Enter link</h1>", unsafe_allow_html=True)
# with st.form("link_form"):
#     link_t = st.text_input("")
#     submit_button = st.form_submit_button("Check")

st.markdown("<h1 style='color: white;font-size: 35px;margin-bottom: 10px;'>Enter link</h1>", unsafe_allow_html=True)

with st.form("link_form"):
    # Use CSS to set the height of the input box
    st.markdown(
        """
        <style>
        input[type="text"] {
            height: 70px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    link_t = st.text_input("", key="link_input")
    submit_button = st.form_submit_button("Check")


if image_file is not None:
    

    # Read the image using OpenCV
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)

    # img = Image.open(io.BytesIO(image_file.read()))

    # # Convert the image to JPEG
    # image_jpeg = img.convert("RGB")
    # with io.BytesIO() as output:
    #     image_jpeg.save(output, format="JPEG")
    #     contents = output.getvalue()
    # Display the image using Streamlit
    st.image(img,channels="BGR")


# # Open a simple image
# img = cv2.imread(image_file)


# converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(
        HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    HSV_result = cv2.bitwise_not(HSV_mask)

    # show results
    img3 = cv2.resize(img, (500, 300))
    # cv2.imshow("1_HSV.jpg",img3)

    img2 = cv2.resize(HSV_result, (500, 300))
    # cv2.imshow("1_HSV.jpg",img2)


    cv2.imwrite("1_HSV.jpg", HSV_result)

    cv2.waitKey(0)

    # im = Image.open(img, mode='r')
    # pix_val = list(im.getdata())
    # pix_val_flat = [x for sets in pix_val for x in sets]
    # print(pix_val_flat)


    # filename = os.path.join(image_file)
    # img = Image.open(filename)
    from pathlib import Path
    string_path = str(os.path.abspath(image_file.name))
    st.write(string_path)
    img = Image.open(string_path)
    width, height = img.size
    Total_pixels = width * height
    print("Dimensions:", img.size, Total_pixels)

    # extracting only black pixels
    number_of_black_pix = np.sum(HSV_result == 0)
    print(number_of_black_pix)

    # count = cv2.countNonZero(img)
    # print(count)

    # im = Image.open('train2.jpg')

    # black = 0
    # red = 0

    # for pixel in im.getdata():
    #     if pixel == (30,144,255): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
    #         blue += 1
    # print('blue=' + str(blue))
    print(number_of_black_pix)
    # white=(number_of_black_pix-Total_pixels)
    # print(white)
    skin_tone_percentage = ((number_of_black_pix/Total_pixels)*100)
    print(skin_tone_percentage)

    print("The skin tone percentage is : ", skin_tone_percentage)

    if skin_tone_percentage > 50:
        # print("This is an explicit image. Block the user!")
        st.write("<p style='color: white; font-size: 30px;'>This is an explicit image!</p>", unsafe_allow_html=True)

    else:
        # print("This is a normal image. No action to be taken on th user")
        st.write("<p style='color: white; font-size: 30px;'>This is a normal image. No action to be taken on the user</p>", unsafe_allow_html=True)
        

    # import imutils
    # img = cv2.imread('color2.jpg')


    # lower_range = np.array([110,50,50])
    # upper_range = np.array([130,255,255])






# Streamlit input txt
# link_t = st.text_area("Enter link")
    
urls_data = pd.read_csv("urldata.csv")


type(urls_data)

# print(urls_data.tail())

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

y = urls_data["label"]

# Features
url_list = urls_data["url"]

vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
#using logistic regression
logit = LogisticRegression()	
logit.fit(X_train, y_train)

# print (url_list.tail())


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)





# Streamlit Input
# link_t = st.text_area("Enter link")

X_predict = [link_t]

X_predict = vectorizer.transform(X_predict)
if link_t:
    New_predict = logit.predict(X_predict)
    print(type(New_predict))
    temp = str(New_predict)

    # st.write(temp)
    
    # if temp == "['malicious']":
    #     st.write(f"<p style='color: red; font-size: 20px'>The URL is: MALICIOUS content</p>", unsafe_allow_html=True)

    if temp == "['malicious']":
        st.write(f"<p style='color: white; font-size: 30px;'>THE URL IS MALICIOUS</p>", unsafe_allow_html=True)
# <h1 style='color: white; font-size: 24px;'>Upload Image</h1>

    elif temp == "['benign']":
     st.write(f"<p style='color: white; font-size: 30px;'>THE URL IS SAFE</p>", unsafe_allow_html=True)






