from models.traditional_model import compute_tfidf_score
from models.gnn_model import gnn_score
from models.rl_model import reinforcement_learning_score

def calculate_final_score(job_desc, cvs):
    """
    Calculate final weighted score combining all AI models.
    :param job_desc: Job description text
    :param cvs: List of CV texts
    :return: List of final scores
    """
    tfidf_scores = compute_tfidf_score(job_desc, cvs)
    gnn_scores = gnn_score(job_desc, cvs)
    rl_scores = reinforcement_learning_score(job_desc, cvs)

    final_scores = (0.4 * tfidf_scores) + (0.3 * gnn_scores) + (0.3 * rl_scores)
    return final_scores
s