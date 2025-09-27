from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from models import Resume, JobDescription


def calculate_tfidf_score(db: Session):
    resumes = db.query(Resume).all()
    job = db.query(JobDescription).first()

    if not job:
        return {"message": "No job description available"}

    vectorizer = TfidfVectorizer()
    contents = [job.description] + [resume.content for resume in resumes]
    vectors = vectorizer.fit_transform(contents)

    for i, resume in enumerate(resumes):
        score = cosine_similarity(vectors[0], vectors[i + 1])[0][0]
        resume.tfidf_score = score

    db.commit()
    return {"message": "TF-IDF scoring completed"}