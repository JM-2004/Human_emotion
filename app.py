import streamlit as st
import numpy as np
import pickle
import re, string
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing import image
from PIL import Image

# Setup
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")
st.title("🤖 Multimodal Emotion Detection")
st.write("Detect emotions from **text and face images** using a fusion of deep learning models.")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
MAXLEN = 229
cv_label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
nlp_label_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
shared_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']  # Fusion labels

# Load resources
@st.cache_resource
def load_nlp_model():
    return load_model("../Models/Emotion_NLP.keras")

@st.cache_resource
def load_cv_model():
    return load_model("../Models/model_CNN.keras")

@st.cache_data
def load_tokenizer():
    with open("../Models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_label_encoder():
    with open("../Models/label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

nlp_model = load_nlp_model()
cv_model = load_cv_model()
tokenizer = load_tokenizer()
le = load_label_encoder()
stop_words = get_stopwords()
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def Removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def Removing_punctuations(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub('\s+', ' ', text).strip()

def Removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = Removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

# --- App UI ---
st.header("🧠 Combined Text + Image Emotion Prediction")

text_input = st.text_area("📝 Enter a sentence:")
image_file = st.file_uploader("🖼️ Upload a face image (48x48 grayscale):", type=["jpg", "jpeg", "png"])

if st.button("🔮 Predict Emotion"):
    if not text_input.strip() or image_file is None:
        st.warning("Please provide **both** text and image.")
    else:
        # --- NLP prediction ---
        norm_text = normalized_sentence(text_input)
        seq = tokenizer.texts_to_sequences([norm_text])
        padded = pad_sequences(seq, maxlen=MAXLEN, truncating='pre')
        nlp_pred = nlp_model.predict(padded)[0]  # shape: (6,)

        # --- CV prediction ---
        img = Image.open(image_file).convert("L")
        img = img.resize((48, 48))
        img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
        cv_pred = cv_model.predict(img_array)[0]  # shape: (7,)

        # --- Extract matching indices ---
        # CV: [Angry=0, Fear=2, Happy=3, Sad=5, Surprise=6]
        # NLP: ['anger'=0, 'fear'=1, 'joy'=2, 'sadness'=4, 'surprise'=5]
        cv_indices = [0, 2, 3, 5, 6]
        nlp_indices = [0, 1, 2, 4, 5]

        cv_filtered = cv_pred[cv_indices]
        nlp_filtered = nlp_pred[nlp_indices]

        # --- Fusion ---
        combined_pred = (cv_filtered + nlp_filtered) / 2
        final_index = np.argmax(combined_pred)
        final_label = shared_labels[final_index]
        confidence = combined_pred[final_index]

        # --- Display ---
        st.success(f"🎯 **Predicted Emotion:** `{final_label}`")
        st.info(f"🧪 **Confidence Score:** `{confidence:.2f}`")

        with st.expander("📊 Show Raw Predictions"):
            st.subheader("Text Model (NLP):")
            for i, label in enumerate(nlp_label_list):
                st.write(f"{label.capitalize()}: {nlp_pred[i]:.2f}")

            st.subheader("Image Model (CV):")
            for i in range(7):
                st.write(f"{cv_label_dict[i]}: {cv_pred[i]:.2f}")

            st.subheader("Fused (Shared 5 Classes):")
            for i, label in enumerate(shared_labels):
                st.write(f"{label}: {combined_pred[i]:.2f}")
