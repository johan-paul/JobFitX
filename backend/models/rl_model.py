import torch
from transformers import BertTokenizer, BertModel

def calculate_learning_score(job_description, resume, candidate_email):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Ensure model is in evaluation mode
    model.eval()

    # Tokenize inputs
    inputs = tokenizer(job_description, resume, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings (using [CLS] token for simplicity)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Calculate cosine similarity
    job_embed = cls_embedding[0]
    resume_embed = cls_embedding[1]

    similarity = torch.nn.functional.cosine_similarity(job_embed.unsqueeze(0), resume_embed.unsqueeze(0)).item()

    # Normalize to 0-100 range
    score = (similarity + 1) * 50

    if score < 50:
        send_rejection_email(candidate_email)

    return round(score, 2)

def send_rejection_email(candidate_email):
    print(f"Sending rejection email to: {candidate_email}")
    # Placeholder for actual email-sending logic
    return True