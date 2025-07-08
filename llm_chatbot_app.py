import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Chatbot LLM", layout="centered")

st.title("💬 LLM Chatbot")
st.caption("Powered by Hugging Face Transformers")

# Créer le modèle de génération de texte
generator = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=150)

# Historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Interface utilisateur
user_input = st.text_input("You:", key="input")

if user_input:
    prompt = f"Human: {user_input}\nAI:"
    response = generator(prompt)[0]['generated_text'].split("AI:")[-1].strip()
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Affichage du chat
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
