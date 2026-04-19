import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Set, Optional, List
import re
import cv2
import speech_recognition as sr
import pyttsx3
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import threading
import queue
import time
import json
import numpy as np

# Set up global queues for communication between threads
transcript_queue = queue.Queue()
audio_queue = queue.Queue()
interview_finished = threading.Event()

class InterviewState:
    def __init__(self, resume_text: str, jd_text: str) -> None:
        self.resume_text: str = resume_text
        self.jd_text: str = jd_text
        self.conversation_history: List[Dict[str, str]] = []
        self.topics_covered: Set[str] = set()
        self.questions_asked: Set[str] = set()
        self.interview_stage: str = "introduction"
        self.user_profile: Dict[str, str] = {}
        self.job_requirements: Dict[str, str] = {}

class InterviewMetrics:
    def __init__(self):
        self.start_time: float = time.time()
        self.end_time: float = 0.0
        self.user_responses: List[Dict] = []
        self.question_count: int = 0
        self.total_words_spoken: int = 0
        self.avg_response_time: float = 0.0

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
    lines = [line.strip("-• \t") for line in jd_text.splitlines() if line.strip()]
    reqs: Dict[str, str] = {}
    for idx, line in enumerate(lines[:50]):
        key = f"req_{idx+1}"
        reqs[key] = line
    return reqs

def find_next_uncovered_technical_topic(jd_text: str, topics_covered: Set[str]) -> Optional[str]:
    candidates: Set[str] = set()
    tech_keywords = [
        "python", "java", "sql", "aws", "gcp", "azure", "docker", "kubernetes",
        "react", "node", "pandas", "spark", "hadoop", "ml", "machine learning",
        "nlp", "transformers", "microservices", "rest", "graphql",
    ]
    for kw in tech_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", jd_text, re.IGNORECASE):
            candidates.add(kw)
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

def test_microphone():
    """Test microphone functionality and provide debugging information."""
    print("=== MICROPHONE TEST ===")
    
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # List available microphones
        print("Available microphones:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"   {index}: {name}")
        
        # Test microphone access
        print("\nTesting microphone access...")
        with sr.Microphone() as source:
            print(f"✓ Microphone found: {source}")
            
            # Test ambient noise adjustment
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            print(f"✓ Energy threshold set to: {r.energy_threshold}")
            
            # Test listening with multiple attempts
            print("\nTesting audio capture...")
            for attempt in range(3):
                print(f"Attempt {attempt + 1}/3: Please say 'Hello, this is a test' now...")
                
                try:
                    audio = r.listen(source, timeout=3, phrase_time_limit=5)
                    print("✓ Audio captured successfully")
                    
                    # Test recognition
                    print("Testing speech recognition...")
                    try:
                        text = r.recognize_google(audio, language='en-US')
                        print(f"✓ Recognition successful: '{text}'")
                        return True
                    except sr.UnknownValueError:
                        print("✗ Could not understand audio - try speaking louder/clearer")
                        if attempt < 2:
                            print("Trying again...")
                            continue
                        return False
                    except sr.RequestError as e:
                        print(f"✗ Recognition service error: {e}")
                        return False
                        
                except sr.WaitTimeoutError:
                    print("✗ No audio detected within timeout")
                    if attempt < 2:
                        print("Trying again...")
                        continue
                    return False
                
    except OSError as e:
        print(f"✗ Microphone error: {e}")
        print("Please check your microphone connection and permissions.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("=== END MICROPHONE TEST ===\n")

# Function to handle Speech-to-Text (STT)
def audio_listener():
    print("=== STARTING AUDIO LISTENER ===")
    
    r = sr.Recognizer()
    
    # Use dynamic energy threshold to auto-adjust to ambient noise
    r.dynamic_energy_threshold = True
    
    try:
        microphone_names = sr.Microphone.list_microphone_names()
        print(f"Available microphones: {microphone_names}")
    except Exception as e:
        print(f"Could not list microphones: {e}")
        return

    working_mic = None
    for mic_index, mic_name in enumerate(microphone_names):
        try:
            with sr.Microphone(device_index=mic_index) as source:
                print(f"Testing access to microphone {mic_index}: {mic_name}")
                working_mic = mic_index
                break
        except Exception as e:
            print(f"✗ Microphone {mic_index} failed: {e}")
            continue

    if working_mic is None:
        print("No working microphone found. Cannot continue.")
        return

    print(f"Using microphone {working_mic}: {microphone_names[working_mic]}")
    
    with sr.Microphone(device_index=working_mic) as source:
        print("Adjusting for ambient noise. Please be silent for a moment...")
        r.adjust_for_ambient_noise(source, duration=1.5)
        print(f"Final energy threshold adjusted to: {r.energy_threshold}")
        
        print("Audio listener started. Listening for speech...")
        
        while not interview_finished.is_set():
            try:
                print("\n🎤 Listening for an answer...")
                
                # Increase timeout and phrase_time_limit for robustness
                audio = r.listen(
                    source, 
                    timeout=15,          # Wait up to 15 seconds for speech to start
                    phrase_time_limit=20 # Allow up to 20 seconds for a full phrase
                )
                
                print("✓ Audio captured! Sending to Google for processing...")
                
                text = r.recognize_google(audio, language='en-US')
                
                if text and text.strip():
                    print(f"You said: '{text.strip()}'")
                    transcript_queue.put(text.strip())
                else:
                    print("No text recognized from audio.")
                    
            except sr.WaitTimeoutError:
                print("⏰ Timeout: No speech detected. Waiting for next turn.")
            except sr.UnknownValueError:
                print("⚠️ Could not understand audio. Try speaking again.")
            except sr.RequestError as e:
                print(f"❌ Could not request results from Google Speech Recognition service; check your internet connection and API status. Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred in audio listener: {e}")
                
    print("Audio listener stopped.")

# Function to handle Text-to-Speech (TTS)
def tts_speaker(audio_queue, finished_event):
    """
    Handles text-to-speech (TTS) conversion in a separate thread.
    Waits for text in the queue and speaks it aloud.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)  # A slightly faster rate for a more natural feel
        engine.setProperty('volume', 0.9)
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        return

    while not finished_event.is_set():
        try:
            # Check the queue for a new message without blocking indefinitely
            text = audio_queue.get(timeout=0.5)
            if text is None:
                print("TTS thread received stop signal.")
                break

            print(f"AI Interviewer: {text}")

            # Queue the text to be spoken
            engine.say(text)
            
            # Use runAndWait to process the speech immediately
            # This is a blocking call that ensures the audio is played
            engine.runAndWait()

            audio_queue.task_done()

        except queue.Empty:
            # This is expected behavior when the queue is empty.
            # We continue the loop to keep the thread alive.
            continue
        except Exception as e:
            print(f"An unexpected error occurred in the TTS speaker: {e}")
            audio_queue.task_done()
            break

    # Final cleanup to ensure any queued speech is processed before the thread exits
    if engine._inLoop:
        engine.endLoop()
    engine.stop()
    print("TTS thread stopped.")


def generate_interview_report(state: InterviewState, metrics: InterviewMetrics, gemini_model: "genai.GenerativeModel") -> str:
    """Generates a comprehensive interview report using the Gemini model."""
    conversation_str = ""
    for i, turn in enumerate(state.conversation_history):
        conversation_str += f"Turn {i+1}: AI asked: {turn['question']}\nUser answered: {turn['answer']}\n\n"

    metrics_str = (
        f"Total Questions Asked: {metrics.question_count}\n"
        f"Total Interview Duration: {metrics.end_time - metrics.start_time:.2f} seconds\n"
        f"Average Response Time: {metrics.avg_response_time:.2f} seconds\n"
        f"Total Words Spoken by User: {metrics.total_words_spoken}\n"
    )

    report_prompt = (
        "You are a professional hiring manager and career counselor. "
        "Analyze the provided interview data and generate a comprehensive report. "
        "The report should include the following sections:\n"
        "1. **Overall Performance Summary**: A brief, high-level summary of the candidate's performance.\n"
        "2. **Technical Skills Assessment**: An evaluation of the candidate's technical knowledge based on their answers, with specific examples. Mention strengths and weaknesses.\n"
        "3. **Behavioral Skills Assessment**: An analysis of the candidate's soft skills (e.g., communication, problem-solving, teamwork) based on their responses. Mention strengths and weaknesses.\n"
        "4. **Areas for Improvement**: Specific, actionable advice for the candidate to improve their interview skills and knowledge.\n"
        "5. **Final Recommendation**: A clear recommendation (e.g., 'Recommend for next round', 'Do not recommend', or 'Further evaluation needed').\n\n"
        "--- Interview Data ---\n"
        f"Job Description: {state.jd_text}\n"
        f"Candidate Resume: {state.resume_text}\n"
        f"Conversation History:\n{conversation_str}\n"
        f"Interview Metrics:\n{metrics_str}\n"
    )

    try:
        response = gemini_model.generate_content(report_prompt)
        report = response.text
        return report
    except Exception as e:
        return f"Error generating report: {e}"

def conduct_interview_with_voice(
    state: InterviewState,
    gemini_model: "genai.GenerativeModel",
    embedding_model: "SentenceTransformer",
    resume_embeddings_dict: Dict[str, "np.ndarray"],
    metrics: InterviewMetrics
) -> None:
    system_prompt = (
        "You are a professional hiring manager. "
        "Interview the candidate based on this job description and resume. "
        f"Job Description: {state.jd_text}\n"
        f"Candidate Resume: {state.resume_text}\n"
    )
    
    chat_session = gemini_model.start_chat(history=[
        {"role": "user", "parts": [system_prompt]}
    ])

    audio_queue.put("Hello, thanks for joining us today. Let's start with your background.")
    audio_queue.join()

    while state.interview_stage != "conclusion":
        prompt: str = ""
        next_topic: Optional[str] = None
        user_answer = ""
        
        if state.interview_stage == "introduction":
            prompt = "Please give a brief introduction about yourself and your career journey so far."
            state.interview_stage = "technical"
        elif state.interview_stage == "technical":
            next_topic = find_next_uncovered_technical_topic(state.jd_text, state.topics_covered)
            if not next_topic or len(state.topics_covered) >= 2:
                state.interview_stage = "behavioral"
                continue
            relevant_context = build_relevant_resume_snippet(next_topic, embedding_model, resume_embeddings_dict)
            prompt = (
                f"Based on this resume context: '{relevant_context}', and the job requirement around '{next_topic}', "
                "ask one specific, insightful technical question that a real hiring manager would ask. Do not simply ask about the topic, ask about a specific application or challenge related to it."
            )
        elif state.interview_stage == "behavioral":
            behavioral_template = find_next_behavioral_question(state.questions_asked)
            if len(state.topics_covered) >= 5:
                state.interview_stage = "conclusion"
                continue
            prompt = (
                "Ask one behavioral question using the STAR method format. "
                f"Prompt the candidate to discuss {behavioral_template} and relate it to their prior roles and experiences from their resume. Do not mention STAR method explicitly in the question."
            )
            next_topic = behavioral_template

        try:
            response = chat_session.send_message(prompt)
            question = response.text or ""
            
            if not question.strip():
                print("Warning: Empty question generated, skipping...")
                continue
                
            audio_queue.put(question)
            audio_queue.join()
            
            response_start_time = time.time()
            print("Listening for your response...")
            
            try:
                user_answer = transcript_queue.get(timeout=30)
            except queue.Empty:
                print("No response received within timeout. Moving to next question.")
                audio_queue.put("I didn't hear a response. Let's move on to the next question.")
                audio_queue.join()
                continue
            
            response_end_time = time.time()
            response_duration = response_end_time - response_start_time
            word_count = len(user_answer.split())
            
            print(f"You: {user_answer}")
            
            metrics.question_count += 1
            metrics.total_words_spoken += word_count
            metrics.user_responses.append({
                "question": question,
                "answer": user_answer,
                "response_time": response_duration,
                "word_count": word_count
            })
            
            state.conversation_history.append({
                "question": question,
                "answer": user_answer,
                "response_time": response_duration,
                "word_count": word_count
            })
            state.questions_asked.add(question)
            if next_topic:
                state.topics_covered.add(next_topic)

            try:
                evaluation_prompt = (
                    f"The last question was: '{question}'. The candidate's response was: '{user_answer}'. "
                    "Evaluate this response for clarity, relevance, and depth. Based on this, please either ask a specific, brief follow-up question to dig deeper on their answer or, if the answer was sufficient, provide a brief transition statement and ask a new, different question from the provided resume/JD. Keep your response concise."
                )
                
                evaluation_response = chat_session.send_message(evaluation_prompt)
                follow_up = evaluation_response.text or ""
                
                if follow_up.strip():
                    audio_queue.put(follow_up)
                    audio_queue.join()
            except Exception as e:
                print(f"Error generating follow-up: {e}")
                continue

        except Exception as e:
            print(f"An error occurred during the interview loop: {e}")
            try:
                audio_queue.put("I'm sorry, I'm experiencing some technical issues. Let's move on.")
                audio_queue.join()
            except:
                pass
            state.interview_stage = "conclusion"
            break

    metrics.end_time = time.time()
    if metrics.question_count > 0:
        metrics.avg_response_time = sum(r['response_time'] for r in metrics.user_responses) / metrics.question_count
    
    audio_queue.put("Great. We will be in touch with you shortly. Thank you.")
    audio_queue.join()
    
    interview_finished.set()
    
    report = generate_interview_report(state, metrics, gemini_model)
    print("\n\n--- INTERVIEW REPORT ---")
    print(report)

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from a resume PDF and load a job description text file."
    )
    parser.add_argument("--resume", required=True, help="Path to the resume PDF file")
    parser.add_argument("--job-description", required=True, help="Path to the job description text file")
    parser.add_argument("--gemini-interview", action="store_true", help="Start an interactive Gemini-powered interview using the loaded context.")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash", help="Gemini model name to use for the interview (default: gemini-1.5-lite).")
    parser.add_argument("--test-microphone", action="store_true", help="Test microphone functionality before starting interview.")
    
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

    if args.test_microphone:
        print("Testing microphone functionality...")
        if not test_microphone():
            print("Microphone test failed. Please check your microphone setup before running the interview.")
            return 1
        print("Microphone test passed! You can proceed with the interview.\n")

    if args.gemini_interview:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            return 1
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(args.gemini_model)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            resume_chunks = text_splitter.split_text(resume_text)
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            resume_embeddings_array = embedding_model.encode(resume_chunks, convert_to_numpy=True, show_progress_bar=False)
            resume_embeddings_dict = dict(zip(resume_chunks, resume_embeddings_array))

            state = InterviewState(resume_text=resume_text, jd_text=job_description_text)
            metrics = InterviewMetrics()

            listener_thread = threading.Thread(target=audio_listener)
            speaker_thread = threading.Thread(target=tts_speaker, args=(audio_queue, interview_finished))
            listener_thread.daemon = True
            speaker_thread.daemon = True
            listener_thread.start()
            speaker_thread.start()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open video stream.")
                return 1

            interview_thread = threading.Thread(target=conduct_interview_with_voice, args=(state, gemini_model, embedding_model, resume_embeddings_dict, metrics))
            interview_thread.start()

            while not interview_finished.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Virtual Interview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            
            listener_thread.join()
            speaker_thread.join()
            interview_thread.join()

        except Exception as exc:
            print(f"Error during Gemini interview: {exc}")
            return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))