import networkx as nx

def calculate_gnn_score(resume_content, job_content):
    """
    Calculate a simple GNN-based score by comparing common words
    between the resume and job description.
    
    Args:
        resume_content (str): Extracted text from the resume.
        job_content (str): Text from the job description.

    Returns:
        float: GNN similarity score (between 0 and 1).
    """
    # Create an undirected graph
    graph = nx.Graph()

    # Tokenize the resume and job description (limited to 50 words each)
    resume_words = resume_content.split()[:50]
    job_words = job_content.split()[:50]

    # Add nodes for resume and job description
    for i, word in enumerate(resume_words):
        graph.add_node(f"resume_{i}", label=word)

    for j, word in enumerate(job_words):
        graph.add_node(f"job_{j}", label=word)

    # Create edges between matching words (word overlap)
    for i, r_word in enumerate(resume_words):
        for j, j_word in enumerate(job_words):
            if r_word == j_word:
                graph.add_edge(f"resume_{i}", f"job_{j}")

    # Score based on edge density (common word connections)
    total_possible_edges = len(resume_words) * len(job_words)
    edge_count = len(graph.edges)

    # Avoid division by zero
    if total_possible_edges == 0:
        return 0.0

    return edge_count / total_possible_edges

# Example usage
if _name_ == "_main_":
    resume_text = "I am skilled in Python, machine learning, and data analysis."
    job_text = "We are looking for someone with experience in Python and data analysis."

    gnn_score = calculate_gnn_score(resume_text, job_text)
    print(f"GNN Score: {gnn_score:.4f}")