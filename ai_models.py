import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.gnn_model import GNNModel
from models.rl_model import rl_scoring





# ---- TF-IDF Scoring (Traditional AI) ----
def tfidf_scoring(job_text, cv_texts):
    vectorizer = TfidfVectorizer()
    all_texts = [job_text] + list(cv_texts.values())
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    job_vector = tfidf_matrix[0]  # First row is the job description
    scores = {
        cv_id: float((tfidf_matrix[i + 1] @ job_vector.T).sum())
        for i, cv_id in enumerate(cv_texts)
    }
    return scores

# ---- GNN Scoring (Graph AI) ----
def gnn_scoring(job_text, cv_texts):
    gnn = GNNModel()  # Load the GNN model
    scores, graph_image = gnn.predict(job_text, cv_texts)  # Get GNN scores and graph visualization
    return scores, graph_image

# ---- RL Scoring (Learning Potential AI) ----
def rl_scoring(job_text, cv_texts):
    rl = RLModel()  # Load the RL model
    scores = rl.predict(job_text, cv_texts)  # Get RL scores
    return scores
