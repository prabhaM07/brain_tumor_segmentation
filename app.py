import streamlit as st
from classifier import app as classifier_app
from chatbot import app as chatbot_app
from Analyser import app as Analyser_app
from main import app as main_app

def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an option:", ["Brain Tumor Classifier", "Brain Tumor Segmentation","Analyser","ChatBot"])

    if option == "Analyser":
        Analyser_app()
    elif option == "ChatBot":
        chatbot_app()
    # Uncomment this if you add the classifier functionality back
    elif option == "Brain Tumor Classifier":
        classifier_app()
    elif option == "Brain Tumor Segmentation":
        main_app()

if __name__ == '__main__':
    main()
