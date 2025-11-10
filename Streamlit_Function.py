from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Multiply, Concatenate, UpSampling2D, Conv2D,Conv2DTranspose,Add,GlobalAveragePooling2D,Dense,BatchNormalization,add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
# model = model_from_json(open("model_part2.json", "r").read())
# model.load_weights("model_second_try.weights.h5")


import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Chroma_VectorStore import chroma_collection
import chromadb.utils.embedding_functions as embedding_functions

# Page configuration
st.set_page_config(
    page_title="Emotion Detection & Analysis",
    page_icon="💁🏻‍♂️",
    layout="wide"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'sentiment_pipeline' not in st.session_state:
    st.session_state.sentiment_pipeline = None

st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password",
    help="Enter your Google Gemini API key"
)

EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model"""
    try:
        model = tf.keras.models.model_from_json(open("Model_Parameters/model_part2.json", "r").read())
        model.load_weights("Model_Parameters/model_second_try.weights.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_sentiment_pipeline():
    """Load the sentiment analysis pipeline"""
    try:
        return pipeline(model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Error loading sentiment pipeline: {str(e)}")
        return None

def preprocess_image(image, target_size=(48, 48)):
    """Preprocess image for model prediction"""
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
        img_resized = cv2.resize(img_gray, target_size)
        img_3channel = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_batch = np.expand_dims(img_3channel, axis=0)
    
    return img_batch

def get_rag_response(emotion, user_query, api_key):
    """Get RAG-based response from Langchain"""
    try:
        google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key
        )
        collection = chroma_collection(google_ef)
        results = collection.query(
            query_texts=[emotion],
            n_results=1
        )
        
        context = results['metadatas'][0][0]['description'] if results['metadatas'] else "No context available"
        
        full_query = f"Detected emotion: {emotion}. User query: {user_query}"
        messages = [
            SystemMessage(
                content=(
                    "You are an empathetic emotional support assistant. "
                    "The user's facial emotion has been detected and they have a query about it. "
                    "Using the following context about the detected emotion, provide a warm, supportive response to the user's query. "
                    "Be compassionate and understanding. "
                    f"\n\nEmotion Context:\n{context}"
                )
            ),
            HumanMessage(content=full_query)
        ]
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def analyze_sentiment(text, emotion_labels):
    """Analyze sentiment of the generated response"""
    try:
        if st.session_state.sentiment_pipeline is None:
            st.session_state.sentiment_pipeline = load_sentiment_pipeline()
        
        if st.session_state.sentiment_pipeline:
            result = st.session_state.sentiment_pipeline(
                text,
                candidate_labels=emotion_labels
            )
            return result
        return None
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return None

st.title("💁🏻‍♂️ Emotion Detection & Support System")
st.markdown("---")

if not api_key:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue.")
    st.stop()

with st.spinner("Loading models..."):
    emotion_model = load_emotion_model()
    if not st.session_state.model_loaded and emotion_model:
        st.session_state.model_loaded = True
        st.success("✅ Models loaded successfully!")

if not emotion_model:
    st.error("❌ Failed to load emotion detection model. Please check model files.")
    st.stop()

st.header("📸 Capture Your Expression")
col1, col2 = st.columns(2)

with col1:
    camera_image = st.camera_input("Take a photo")

with col2:
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)
        
        if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing emotion..."):
                processed_img = preprocess_image(image)
                predictions = emotion_model.predict(processed_img, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                detected_emotion = EMOTION_LABELS[predicted_class]
                
                st.session_state.detected_emotion = detected_emotion
                st.session_state.confidence = confidence
                st.session_state.predictions = predictions[0]
                st.session_state.image_analyzed = True

if hasattr(st.session_state, 'detected_emotion') and st.session_state.image_analyzed:
    st.markdown("---")
    st.header("📊 Analysis Results")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detected Emotion", st.session_state.detected_emotion)
    
    with col2:
        st.metric("Confidence", f"{st.session_state.confidence * 100:.2f}%")
    
    with col3:
        st.metric("Status", "✅ Analyzed")
    
    with st.expander("📈 View All Emotion Probabilities"):
        for i, emotion in enumerate(EMOTION_LABELS):
            st.progress(float(st.session_state.predictions[i]), text=f"{emotion}: {st.session_state.predictions[i]*100:.2f}%")
    

    st.markdown("---")
    st.header("💬 Ask About Your Emotion")
    
    user_query = st.chat_input(
        placeholder="Describe the image or ask something about your detected emotion..."
    )
    
    if user_query:
        st.session_state.user_query = user_query
        st.session_state.query_submitted = True
    
    if hasattr(st.session_state, 'query_submitted') and st.session_state.query_submitted:
        
        with st.chat_message("user"):
            st.write(st.session_state.user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating personalized response..."):
                response_text = get_rag_response(
                    st.session_state.detected_emotion, 
                    st.session_state.user_query, 
                    api_key
                )
                st.write(response_text)
                st.session_state.response_text = response_text
    
    if hasattr(st.session_state, 'response_text'):
        st.markdown("---")
        st.header("🎯 Sentiment Analysis Report")
        
        with st.spinner("Analyzing response sentiment..."):
            sentiment_result = analyze_sentiment(
                st.session_state.response_text,
                [label.lower() for label in EMOTION_LABELS]
            )
            
            if sentiment_result:
                
                st.subheader("Sentiment Distribution")
                
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    
                    import pandas as pd
                    df = pd.DataFrame({
                        'Emotion': [label.capitalize() for label in sentiment_result['labels']],
                        'Score': sentiment_result['scores']
                    })
                    st.bar_chart(df.set_index('Emotion'))
                
                with col2:
                    
                    st.subheader("Top Sentiments")
                    for i in range(min(3, len(sentiment_result['labels']))):
                        st.metric(
                            sentiment_result['labels'][i].capitalize(),
                            f"{sentiment_result['scores'][i]*100:.1f}%"
                        )
                
                
                with st.expander("📋 Detailed Sentiment Scores"):
                    for label, score in zip(sentiment_result['labels'], sentiment_result['scores']):
                        st.write(f"**{label.capitalize()}:** {score*100:.2f}%")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit, TensorFlow, and LangChain"
    "</div>",
    unsafe_allow_html=True
)
