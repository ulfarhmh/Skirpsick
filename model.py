import streamlit as st
from transformers import pipeline
import base64

# Function to get base64 of the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get the base64 of the image
base64_background = get_base64_of_bin_file('rrrr.jpg')

# Streamlit app
st.set_page_config(layout="centered")  # Centered layout

# Adding custom CSS for background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_background}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, p {{
        color: #ffffff;  /* Text color for title and description */
        text-align: center; /* Center align text */
    }}
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: white;
        text-align: center;
        padding: 10px;
    }}
    .small-title {{
        font-size: 24px;
    }}
    .prediction-container {{
        text-align: left;
        margin-left: 20px;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .progress-container {{
        width: 80%;
        text-align: left;
        margin-left: 20px;
        margin-top: 5px;
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered title and description
st.markdown("<h1 style='text-align: center;'>Emotion Prediction<br>using NRC Lexicon</h1>", unsafe_allow_html=True)
st.markdown("""
            <p style='text-align: center;'>This app predicts 8 emotions based on LSTM model with NRC Emotion Lexicon.</p>
            """, unsafe_allow_html=True)

# Text input
input_text = st.text_area('Enter text:', '')

# Emoticons for emotions
emotion_emoticons = {
    "anger": "😠",
    "anticipation": "🤔",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "sadness": "😢",
    "surprise": "😲",
    "trust": "🤝"
}

# Prediction button
if st.button('Analyze'):
    if input_text:
        # Display input text with left alignment
        st.markdown(f"<p style='text-align: left;'>Input text: {input_text}</p>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: left;'>Predicted Emotions:</h3>", unsafe_allow_html=True)

        # Initialize pipeline for text classification using the NusaBERT model
        classifier = pipeline("text-classification", model="Aardiiiiy/NusaBERT-base-Indonesian-Plutchik-emotion-analysis")

        # Perform prediction
        predictions = classifier([input_text], return_all_scores=True)

        # Extract predicted labels and scores
        sorted_predictions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
        top_two_predictions = sorted_predictions[:2]

        # Display top two predicted emotions
        for prediction in top_two_predictions:
            label = prediction['label']
            score = prediction['score'] * 100
            emoticon = emotion_emoticons.get(label.lower(), '❓')  # Default emoticon if not found
            st.markdown(f"<div class='prediction-container'>{label.capitalize()} {emoticon} {score:.2f}%</div>", unsafe_allow_html=True)
            st.progress(int(score))

    else:
        st.write('Please enter some text to analyze.')

# Footer
st.markdown("<div class='footer'>@ulfarahmah</div>", unsafe_allow_html=True)