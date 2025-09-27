from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)
    skills = Column(Text, nullable=True)  # Comma-separated skills
    experience = Column(Text, nullable=True)  # Experience details
    education = Column(Text, nullable=True)  # Education details
    summary = Column(Text, nullable=True)  # Personal summary

    # Relationship with JobMatchingScore (if needed)
    scores = relationship("JobMatchingScore", back_populates="resume")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(Text, nullable=True)  # Comma-separated skills
    company_name = Column(String, nullable=True)
    location = Column(String, nullable=True)

    # Relationship with JobMatchingScore (if needed)
    scores = relationship("JobMatchingScore", back_populates="job")


class JobMatchingScore(Base):
    __tablename__ = "job_matching_scores"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    job_id = Column(Integer, ForeignKey("job_descriptions.id"))
    tfidf_score = Column(Integer, nullable=True)  # Traditional Matching
    graph_ai_score = Column(Integer, nullable=True)  # GNN Score
    learning_potential_score = Column(Integer, nullable=True)  # RL Score

    # Relationships
    resume = relationship("Resume", back_populates="scores")
    job = relationship("JobDescription", back_populates="scores")
