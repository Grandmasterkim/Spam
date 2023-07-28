import streamlit as st
import pickle

from PIL import Image


model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))



def main():
    st.title("Spam Mail Detector")
    st.subheader('By Victor O')
    image = Image.open('logo.jpg')
    st.image(image, caption='Spam image')
    msg = st.text_input("Paste or type your text to predict: ")
    if st.button('CHECK'):
        data = [msg]
        vect=cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1:
            st.error("This is a spam mail")
        else:
            st.success("This is not a spam mail")




main()