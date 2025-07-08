# NLP2


- Topic Modeling:

We applied **two methods** to discover 10 topics:

 1. Latent Dirichlet Allocation (LDA)

* Vectorization method: **CountVectorizer**
* Generates probabilistic topic distributions
* Topics tend to be more **general**, mixing themes found across multiple documents

 2. Non-negative Matrix Factorization (NMF)

* Vectorization method: **TF-IDF**
* Based on linear algebra and decomposition
* Produces **more specific**, sharply distinct topics

 🔝 Top 10 Words per Topic

Barplots were generated for each model to visualize the 10 most representative words in each topic.

These provide insight into the themes captured:

* **LDA** topics tend to overlap slightly but reflect major discussion areas (e.g., religion, technology, politics).
* **NMF** topics show tighter clustering around more distinct vocabulary.

---

## ☁️ Word Clouds

We generated wordclouds for each method using the most frequent words across all topics.

* **LDA WordCloud**: Broader topics with more common words
* **NMF WordCloud**: More refined words that reflect document-specific terms

---

## ⚙️ Model Performance & Comments

* Both models performed reasonably well on a reduced vocabulary (1000 words)
* **LDA** can be more time-consuming due to its probabilistic nature
* **NMF** is faster and often more interpretable when using TF-IDF


- Sentiment Analysis on Tweets Dataset

We applied two different sentiment analysis methods on a sample of 500 tweets.

## 1️⃣ VADER Algorithm (Rule-based)

- Tool: `SentimentIntensityAnalyzer` from NLTK
- Output:
  - `vader_sentiment`: Sentiment label (`positive`, `neutral`, `negative`)
  - `vader_score`: Compound sentiment score between -1 and 1

### ✅ Pros:
- Fast and efficient for short texts (like tweets)
- No model training required

### ⚠️ Cons:
- Limited to rule-based patterns
- Less accurate on sarcasm, idioms, or context-dependent expressions

---

## 2️⃣ Transformers Model (Hugging Face Pipeline)

- Model: Pretrained Transformer (`distilbert-base-uncased-finetuned-sst-2-english`)
- Output:
  - `transformer_sentiment`: Sentiment label (`positive`, `negative`)
  - `transformer_score`: Confidence score from 0 to 1

### ✅ Pros:
- Context-aware
- Higher accuracy on complex linguistic structures

### ⚠️ Cons:
- Slower inference
- Requires GPU/CPU resources

---

## 🧾 Result Columns Added to the Dataset

| Column Name            | Description                          |
|------------------------|--------------------------------------|
| cleaned_tweet          | Cleaned and preprocessed tweet       |
| vader_sentiment        | Sentiment label (from VADER)         |
| vader_score            | Sentiment score (from VADER)         |
| transformer_sentiment  | Sentiment label (from Transformers)  |
| transformer_score      | Sentiment score (from Transformers)  |

---

## 💡 Final Notes

- Both methods are complementary: VADER is good for speed, Transformers are better for accuracy.
- Further exploration could include: emotion classification, sarcasm detection, or multilingual models.


- Email Classification with LLM :

We used a Hugging Face LLM (`google/flan-t5-base`) to classify emails from the `spam.csv` dataset as **spam** or **ham** using prompt engineering.

### 📊 Output Example

| Email Text                       | True Label | LLM Prediction |
|----------------------------------|------------|----------------|
| Win a FREE cruise to Bahamas!   | spam       | spam           |
| Are we still on for tomorrow?   | ham        | ham            |
| Claim your $1000 gift card now! | spam       | spam           |


### 📈 Evaluation (Sample-Based)

We compared the model predictions to true labels using a small evaluation set.

**Metrics:**

- ✅ Accuracy: 100% (on 10 samples)
- ⚠️ Note: Results may vary slightly due to the generative nature of LLMs

---

### ✅ Advantages

- No training required
- Great for fast prototyping
- Easy to swap LLMs (Mistral, Phi, Zephyr, etc.)

### ⚠️ Limitations

- Slower than traditional models (especially on large datasets)
- May hallucinate or return inconsistent format if prompt is unclear
- Needs careful prompting to ensure predictable outputs


## 💬 Chatbot using LLM and Streamlit

We developed a simple conversational **chatbot** using a **Large Language Model (LLM)** from Hugging Face and deployed it as a **Streamlit app**.

---

### ⚙️ Tools Used

- `streamlit` – for the web app interface
- `transformers` – to load and use a pre-trained text generation model
- LLM: `google/flan-t5-base` (instruction-tuned)

---

### 🧠 How It Works

- The user types a question or message into the chat interface.
- The model receives a prompt like:

- Example:
You: What is the capital of Denmark?
Bot: The capital of Denmark is Copenhagen.

✅ Strengths
Simple and fast to deploy
Instruction-following model provides structured answers
Streamlit allows rapid prototyping of UI

⚠️ Limitations
No conversation memory (stateless)
The model may generate incomplete or vague answers
For long contexts or advanced features, memory/history must be engineered
