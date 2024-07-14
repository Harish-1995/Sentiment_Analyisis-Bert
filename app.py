import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
stop_words = set(stopwords.words('english'))

nltk.download('stopwords')
nltk.download('wordnet')


# regex_standardize
def regex_standardize(text):
    url_pattern = r'\b(?:https?://|www\:.)[^\s]+\b' 
    text = re.sub(url_pattern, '', text)
    username_tag_pattern = r'@\S+'
    text = re.sub(username_tag_pattern, '', text)
    punc_removing = r'[^a-z A-Z 0-9]'
    text = re.sub(punc_removing, '', text)
    return text

lemmatizer = WordNetLemmatizer()

# Text Processing 
def preprocessing_pipeline(text):
    # text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in text.split()])
    return text

# Pretrained_model
from sentence_transformers import SentenceTransformer
pretrained_model = SentenceTransformer('distilbert/distilbert-base-uncased-finetuned-sst-2-english')

# Bert word embedding
def bert_embedding(text):
    return pretrained_model.encode(text)

#loading model and label encoder pickle file
with open("random_search_model.pkl", "rb") as file:
    rfc_model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file_1:
    label_encoder = pickle.load(file_1)


#steamlit app
st.title("Emotion Analysis")

user_name = st.text_input(" 🙋‍♂️ Enter your name my friend")

user_input = st.text_input("Enter your emotion's here")

def prediction_pipeline(text):
    text = regex_standardize(text)
    text = preprocessing_pipeline(text)
    text = bert_embedding(text)
    predicted_emotions = rfc_model.predict([text])
    emotion_decoded = label_encoder.inverse_transform(predicted_emotions)
    return emotion_decoded

prediction = prediction_pipeline(user_input)
# results = st.text(f"your emotion seems to be: {prediction[0]}")

if st.button('submit'):
    if prediction[0] == 'positive':
        st.success(f"{user_name} your emotion seems to be: {prediction[0]} 😎")
    elif prediction[0] =='neutral':
        st.warning(f"{user_name} your emotion seems to be: {prediction[0]} 😕")
    else:
         st.error(f"{user_name} your emotion seems to be: {prediction[0]} 😞")

