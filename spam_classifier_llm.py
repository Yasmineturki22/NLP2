import pandas as pd
from transformers import pipeline

# Charger les données
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
df = df[['text', 'target']].rename(columns={"text": "text", "target": "label_true"})
df = df.sample(10, random_state=42)  # Échantillon rapide

# Charger le modèle LLM avec le bon pipeline
classifier = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=10)

# Fonction de prompt + réponse propre
def classify_email_llm(text):
    prompt = f"Classify this email as 'spam' or 'ham': {text}"
    try:
        result = classifier(prompt)[0]['generated_text']
        return result.strip().lower()
    except Exception as e:
        print("Error:", e)
        return "error"

# Appliquer la classification LLM
df['llm_predicted'] = df['text'].apply(classify_email_llm)

# Affichage et sauvegarde
print(df[['text', 'label_true', 'llm_predicted']])
df.to_csv("spam_llm_classified.csv", index=False)
