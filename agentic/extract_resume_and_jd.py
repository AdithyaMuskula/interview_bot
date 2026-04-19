import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Set, Optional, List
import re

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


class InterviewState:

    def __init__(self, resume_text: str, jd_text: str) -> None:
        self.resume_text: str = resume_text
        self.jd_text: str = jd_text
        self.conversation_history: List[Dict[str, str]] = []
        self.topics_covered: Set[str] = set()
        self.questions_asked: Set[str] = set()
        self.interview_stage: str = "introduction"  # introduction | technical | behavioral | conclusion
        self.user_profile: Dict[str, str] = {}  # facts extracted from resume
        self.job_requirements: Dict[str, str] = {}  # requirements extracted from JD


def extract_text_from_pdf(pdf_path: Path) -> str:

    text = ""
    with pdf_path.open("rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text


def read_text_file(text_path: Path) -> str:

    return text_path.read_text(encoding="utf-8", errors="ignore")


def extract_key_facts_from_resume(resume_text: str) -> Dict[str, str]:

    facts: Dict[str, str] = {}
    # Naive extractions as placeholders (email, phone, years of experience)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text)
    if email_match:
        facts["email"] = email_match.group(0)
    phone_match = re.search(r"\+?\d[\d\s().-]{8,}\d", resume_text)
    if phone_match:
        facts["phone"] = phone_match.group(0)
    yrs_match = re.search(r"(\d+)[+\s-]*(?:years|yrs)\s+of\s+experience", resume_text, re.IGNORECASE)
    if yrs_match:
        facts["years_experience"] = yrs_match.group(1)
    return facts


def extract_requirements_from_jd(jd_text: str) -> Dict[str, str]:

    # Very simple keyword collection by line; can be replaced with LLM/RAG later
    lines = [line.strip("-• \t") for line in jd_text.splitlines() if line.strip()]
    reqs: Dict[str, str] = {}
    for idx, line in enumerate(lines[:50]):
        key = f"req_{idx+1}"
        reqs[key] = line
    return reqs


def find_next_uncovered_technical_topic(jd_text: str, topics_covered: Set[str]) -> Optional[str]:

    # Tokenize JD into candidate topics (naive: capitalized words/phrases and tech keywords)
    candidates: Set[str] = set()
    tech_keywords = [
        "python", "java", "sql", "aws", "gcp", "azure", "docker", "kubernetes",
        "react", "node", "pandas", "spark", "hadoop", "ml", "machine learning",
        "nlp", "transformers", "microservices", "rest", "graphql",
    ]
    for kw in tech_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", jd_text, re.IGNORECASE):
            candidates.add(kw)
    # Prefer candidates not yet covered
    for topic in candidates:
        if topic not in topics_covered:
            return topic
    return None


def find_next_behavioral_question(asked: Set[str]) -> str:

    templates = [
        "a challenging problem you solved under a tight deadline",
        "a time you handled a conflict within your team",
        "a situation where you led without formal authority",
        "a failure you learned from and how you improved",
        "how you prioritized multiple high-impact tasks",
    ]
    for tmpl in templates:
        if tmpl not in asked:
            return tmpl
    return templates[-1]


def build_relevant_resume_snippet(
    query: str,
    embedding_model: "SentenceTransformer",
    resume_embeddings_lookup: Dict[str, "np.ndarray"],
) -> str:

    try:
        jd_embedding = embedding_model.encode(query)
    except Exception:
        return ""
    similarities: Dict[str, float] = {}
    for resume_chunk, resume_embedding in resume_embeddings_lookup.items():
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        sim = _cos(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
        similarities[resume_chunk] = float(sim)
    top_snippets = [t for t, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]]
    return " ".join(top_snippets)


def conduct_interview(
    state: InterviewState,
    gemini_model: "genai.GenerativeModel",
    embedding_model: "SentenceTransformer",
    resume_embeddings_dict: Dict[str, "np.ndarray"],
) -> None:

    system_prompt = (
        "You are a professional hiring manager. "
        "Interview the candidate based on this job description and resume. "
        f"Job Description: {state.jd_text}\n"
        f"Candidate Resume: {state.resume_text}\n"
        f"Current Stage: {state.interview_stage}."
    )
    chat_session = gemini_model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])

    print("AI Interviewer: Hello, thanks for joining us today. Let's start with your background.")

    while state.interview_stage != "conclusion":
        prompt: str = ""
        next_topic: Optional[str] = None

        if state.interview_stage == "introduction":
            prompt = "Please give a brief introduction about yourself and your career journey so far."
            state.interview_stage = "technical"

        elif state.interview_stage == "technical":
            next_topic = find_next_uncovered_technical_topic(state.jd_text, state.topics_covered)
            if not next_topic:
                state.interview_stage = "behavioral"
                continue
            relevant_context = build_relevant_resume_snippet(next_topic, embedding_model, resume_embeddings_dict)
            prompt = (
                f"Based on this resume context: '{relevant_context}', and the job requirement around '{next_topic}', "
                "ask one specific technical question."
            )

        elif state.interview_stage == "behavioral":
            behavioral_template = find_next_behavioral_question(state.questions_asked)
            prompt = (
                "Ask one behavioral question using the STAR method cues. "
                f"Prompt the candidate to discuss {behavioral_template} and relate it to their prior roles."
            )
            next_topic = behavioral_template
            if len(state.topics_covered) >= 3:
                state.interview_stage = "conclusion"

        elif state.interview_stage == "conclusion":
            prompt = "Thank you for your time today. Do you have any questions about the role or company?"

        response = chat_session.send_message(prompt)
        question = response.text or ""
        print("AI Interviewer: " + question)

        user_answer = input("You: ")
        state.conversation_history.append({"role": "user", "content": user_answer})
        state.questions_asked.add(question)
        if next_topic:
            state.topics_covered.add(next_topic)

    print("AI Interviewer: Great. We will be in touch with you shortly. Thank you.")


def main(argv: list[str]) -> int:

    parser = argparse.ArgumentParser(
        description="Extract text from a resume PDF and load a job description text file."
    )
    parser.add_argument(
        "--resume",
        required=True,
        help="Path to the resume PDF file",
    )
    parser.add_argument(
        "--job-description",
        required=True,
        help="Path to the job description text file",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write combined output (resume + JD) as UTF-8 text.",
    )
    parser.add_argument(
        "--gemini-interview",
        action="store_true",
        help="Start an interactive Gemini-powered interview using the loaded context.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model name to use for the interview (default: gemini-1.5-lite).",
    )

    args = parser.parse_args(argv)

    resume_path = Path(args.resume)
    jd_path = Path(args.job_description)

    if not resume_path.exists():
        print(f"Error: resume file not found: {resume_path}")
        return 1
    if not jd_path.exists():
        print(f"Error: job description file not found: {jd_path}")
        return 1

    try:
        resume_text = extract_text_from_pdf(resume_path)
    except Exception as exc:
        print(f"Error reading PDF: {exc}")
        return 1

    try:
        job_description_text = read_text_file(jd_path)
    except Exception as exc:
        print(f"Error reading job description: {exc}")
        return 1

    print("Resume text loaded.")
    print("Job description text loaded.")

    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    resume_chunks = text_splitter.split_text(resume_text)
    jd_chunks = text_splitter.split_text(job_description_text)
    print(f"Split resume into {len(resume_chunks)} chunks.")
    print(f"Split job description into {len(jd_chunks)} chunks.")

    # Build embeddings for chunks
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    resume_embeddings_array = embedding_model.encode(
        resume_chunks, convert_to_numpy=True, show_progress_bar=False
    )
    jd_embeddings_array = embedding_model.encode(
        jd_chunks, convert_to_numpy=True, show_progress_bar=False
    )
    resume_embeddings_dict = dict(zip(resume_chunks, resume_embeddings_array))
    jd_embeddings_dict = dict(zip(jd_chunks, jd_embeddings_array))
    print("Generated embeddings for all text chunks.")

    # Helper to retrieve most relevant resume snippets for a given JD chunk
    def get_most_relevant_resume_snippet(jd_chunk: str, resume_embeddings_lookup: dict[str, "np.ndarray"]) -> str:
        jd_embedding = embedding_model.encode(jd_chunk)
        similarities: dict[str, float] = {}
        for resume_chunk, resume_embedding in resume_embeddings_lookup.items():
            # sklearn expects 2D arrays
            similarity = cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
            similarities[resume_chunk] = similarity

        # Top 3 snippets by similarity
        sorted_snippets = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        top_snippets = [snippet for snippet, _ in sorted_snippets[:3]]
        return " ".join(top_snippets)

    # Example usage (can be removed or replaced with CLI arg later)
    jd_skills_chunk = (
        "Candidate must have strong experience with Python, SQL, and cloud services (AWS, GCP)."
    )
    relevant_resume_info = get_most_relevant_resume_snippet(jd_skills_chunk, resume_embeddings_dict)
    print(
        "Most relevant resume snippets for the skills section:\n" + relevant_resume_info
    )

    if args.out:
        out_path = Path(args.out)
        try:
            combined = (
                "=== RESUME TEXT ===\n" + resume_text + "\n\n=== JOB DESCRIPTION ===\n" + job_description_text
            )
            out_path.write_text(combined, encoding="utf-8")
            print(f"Combined output written to: {out_path}")
        except Exception as exc:
            print(f"Error writing output file: {exc}")
            return 1

    # Optional Gemini-powered interview
    if args.gemini_interview:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            return 1
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(args.gemini_model)

            # Initialize interview state and enrich with extracted info
            state = InterviewState(resume_text=resume_text, jd_text=job_description_text)
            state.user_profile = extract_key_facts_from_resume(resume_text)
            state.job_requirements = extract_requirements_from_jd(job_description_text)

            conduct_interview(
                state=state,
                gemini_model=gemini_model,
                embedding_model=embedding_model,
                resume_embeddings_dict=resume_embeddings_dict,
            )
        except Exception as exc:
            print(f"Error during Gemini interview: {exc}")
            return 1

    return 0


if __name__ == "__main__":

    raise SystemExit(main(sys.argv[1:]))


