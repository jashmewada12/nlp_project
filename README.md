# 📬 Automated Customer Support Ticket Router

An end-to-end Natural Language Processing (NLP) pipeline designed to automatically classify, process, and route customer support emails and tickets to the appropriate department using custom text embeddings, machine learning classifiers, and state-of-the-art agent workflows.

---

## 🚀 Features

- **Advanced Text Preprocessing**: Custom cleaning pipelines incorporating tokenization, stop-word removal, and text normalization.
- **Vector Embeddings**: Custom-trained TF-IDF and Word2Vec embedding models to capture semantic context from support messages.
- **Automated Ticket Classification**: ML-powered categorization to route incoming issues to correct functional departments (e.g., Technical Support, Billing, General Inquiries).
- **LangGraph Workflow Integration**: Orchestrated node-based graph workflow to handle end-to-end decision-making and routing steps.
- **Gmail API Integration**: Seamlessly pulls incoming tickets, processes them through the NLP pipeline, and applies automated responses or labeling.

---

## 🛠️ Tech Stack

- **Language:** Python
- **NLP / ML:** Scikit-learn, Gensim (Word2Vec), NLTK / SpaCy
- **Orchestration:** LangGraph
- **APIs & Integrations:** Gmail API
- **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
nlp_project/
│
├── data/                  # Raw and processed dataset files
├── models/                # Saved TF-IDF, Word2Vec, and classifier models
├── src/                   # Core source code
│   ├── preprocessing.py   # Text cleaning and tokenization logic
│   ├── embeddings.py      # TF-IDF & Word2Vec embedding generators
│   ├── classifier.py      # Ticket classification models
│   └── workflow.py        # LangGraph agent routing definitions
│
├── .gitignore
├── requirements.txt       # Project dependencies
└── README.md
