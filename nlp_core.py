import re
import math
from collections import Counter

# ==========================================
# 1. PREPROCESSING PIPELINE
# ==========================================
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "she", 
    "it", "they", "the", "a", "an", "is", "was", "are", "were", "be", "have", 
    "has", "had", "do", "does", "did", "to", "from", "in", "on", "at", "by", 
    "for", "with", "about", "against", "into", "through", "during", "before", 
    "after", "above", "below", "up", "down", "in", "out", "off", "over", "under", 
    "again", "further", "then", "once", "here", "there", "when", "where", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", 
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
    "can", "will", "just", "don", "should", "now", "this", "but", "and"
}

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()

def simple_stemmer(word):
    suffixes = ('ing', 'ly', 'ed', 'ious', 'ies', 'es', 's', 'ment')
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

def preprocess_text(text):
    tokens = clean_and_tokenize(text)
    return [simple_stemmer(w) for w in tokens if w not in STOPWORDS and len(w) > 1]


# ==========================================
# 2. SCRATCH TF-IDF VECTORIZER
# ==========================================
class ScratchTfidfVectorizer:
    def __init__(self):
        self.vocab = {}          
        self.idf = {}            
        self.doc_count = 0

    def fit(self, tokenized_docs):
        self.doc_count = len(tokenized_docs)
        doc_freq = Counter()

        # Count how many documents contain each word
        for doc in tokenized_docs:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Calculate IDF (Inverse Document Frequency)
        idx = 0
        for token, df in doc_freq.items():
            self.vocab[token] = idx
            self.idf[token] = math.log((1.0 + self.doc_count) / (1.0 + df)) + 1.0
            idx += 1

    def transform(self, tokenized_docs):
        vectors = []
        vocab_size = len(self.vocab)

        for doc in tokenized_docs:
            doc_len = len(doc)
            tf_counts = Counter(doc)
            vec = [0.0] * vocab_size

            if doc_len == 0:
                vectors.append(vec)
                continue

            # Calculate TF-IDF weight for each word
            for token, count in tf_counts.items():
                if token in self.vocab:
                    index = self.vocab[token]
                    tf = count / doc_len
                    vec[index] = tf * self.idf[token]

            # L2 Normalization (Scale vector to have a length of 1)
            norm = math.sqrt(sum(val ** 2 for val in vec))
            if norm > 0:
                vec = [val / norm for val in vec]

            vectors.append(vec)
        return vectors

    def fit_transform(self, tokenized_docs):
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)


# ==========================================
# 3. SCRATCH NAIVE BAYES CLASSIFIER
# ==========================================
class ScratchMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace Smoothing parameter
        self.classes = []
        self.class_priors = {}
        self.feature_log_probs = {} 

    def fit(self, X, y):
        self.classes = list(set(y))
        total_docs = len(y)
        num_features = len(X[0])
        
        # Calculate Class Priors P(c)
        class_counts = Counter(y)
        for c in self.classes:
            self.class_priors[c] = math.log(class_counts[c] / total_docs)

        # Calculate Likelihoods P(w|c) with Laplace Smoothing
        for c in self.classes:
            feature_totals = [0.0] * num_features
            for i in range(total_docs):
                if y[i] == c:
                    for j in range(num_features):
                        feature_totals[j] += X[i][j]

            total_word_mass = sum(feature_totals)
            smoothed_denominator = total_word_mass + (self.alpha * num_features)

            log_probs = [
                math.log((feature_totals[j] + self.alpha) / smoothed_denominator)
                for j in range(num_features)
            ]
            self.feature_log_probs[c] = log_probs

    def predict_single(self, vector):
        best_score = -float('inf')
        best_class = None

        # Calculate Score(c) = log P(c) + sum(log P(w|c) * weight)
        for c in self.classes:
            score = self.class_priors[c]
            for j, weight in enumerate(vector):
                if weight > 0:
                    score += weight * self.feature_log_probs[c][j]

            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    def predict(self, X):
        return [self.predict_single(vec) for vec in X]