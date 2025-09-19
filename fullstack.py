# Install required packages
!pip install pydub noisereduce pyannote.audio transformers torchaudio librosa openai-whisper --quiet

# Imports
import os
import torch
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import librosa
from pyannote.audio import Pipeline
import whisper
from transformers import pipeline

# --- 1. Load and enhance audio ---
def enhance_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
    librosa.output.write_wav(output_path, reduced_noise, sr=audio.frame_rate)
    return output_path

# --- 2. Speaker diarization ---
def diarize_audio(audio_path):
    # Use pyannote pretrained pipeline (requires huggingface token)
    # You need to set your HF_TOKEN as environment variable or insert here
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set your Huggingface token in Colab env variables
    pipeline_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_TOKEN)
    diarization = pipeline_diarization(audio_path)
    return diarization

# --- 3. Transcribe audio with Whisper ---
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

# --- 4. Analyze diarization and transcription to get talk-time ratio and longest monologue ---
def analyze_diarization(diarization):
    speaker_times = {}
    longest_monologue = {"speaker": None, "duration": 0}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
        if duration > longest_monologue["duration"]:
            longest_monologue = {"speaker": speaker, "duration": duration}
    total_time = sum(speaker_times.values())
    talk_time_ratio = {spk: round((dur / total_time) * 100, 2) for spk, dur in speaker_times.items()}
    return talk_time_ratio, longest_monologue

# --- 5. Count questions per speaker ---
def count_questions(transcript_segments):
    question_words = {"who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "would", "will", "shall"}
    questions_per_speaker = {}
    total_questions = 0
    for seg in transcript_segments:
        speaker = seg["speaker"]
        text = seg["text"].lower()
        # Count question marks
        q_marks = text.count("?")
        # Count interrogative words at start of sentences
        q_words = sum(text.startswith(w + " ") for w in question_words)
        q_count = q_marks + q_words
        questions_per_speaker[speaker] = questions_per_speaker.get(speaker, 0) + q_count
        total_questions += q_count
    return questions_per_speaker, total_questions

# --- 6. Sentiment analysis ---
def analyze_sentiment(text):
    sentiment_classifier = pipeline("sentiment-analysis")
    result = sentiment_classifier(text)
    return result[0]["label"]

# --- 7. Extract actionable insight ---
def extract_insight(text):
    zero_shot = pipeline("zero-shot-classification")
    candidate_labels = ["follow up", "product interest", "pricing concern", "schedule demo", "no interest", "technical issue", "customer complaint"]
    result = zero_shot(text, candidate_labels)
    return result["labels"][0]

# --- 8. Identify sales rep vs customer ---
def identify_roles(talk_time_ratio, questions_per_speaker, transcript_segments):
    # Heuristic: speaker with more questions and more talk time is sales rep
    speakers = list(talk_time_ratio.keys())
    if len(speakers) < 2:
        return {"sales_rep": speakers[0], "customer": None}
    spk1, spk2 = speakers[0], speakers[1]
    score1 = talk_time_ratio.get(spk1, 0) + questions_per_speaker.get(spk1, 0)*5
    score2 = talk_time_ratio.get(spk2, 0) + questions_per_speaker.get(spk2, 0)*5
    if score1 >= score2:
        return {"sales_rep": spk1, "customer": spk2}
    else:
        return {"sales_rep": spk2, "customer": spk1}

# --- Main pipeline ---
def analyze_call(audio_file):
    enhanced_path = "enhanced_call.wav"
    enhance_audio(audio_file, enhanced_path)

    diarization = diarize_audio(enhanced_path)

    # Convert diarization to segments for question counting
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    # Transcribe full audio
    transcription_result = transcribe_audio(enhanced_path)
    full_text = transcription_result["text"]

    # Map transcript segments to diarization segments approximately
    # Whisper does not provide speaker labels, so we approximate by time alignment
    # Here we split transcript into chunks by time and assign speaker by diarization
    # For simplicity, assign all text to one speaker (can be improved with forced alignment)
    # We'll just assign full text to speaker with max talk time for question count heuristic
    talk_time_ratio, longest_monologue = analyze_diarization(diarization)

    # For question count, split transcript by sentences and assign to speaker with max talk time
    import re
    sentences = re.split(r'(?<=[.?!])\s+', full_text)
    max_speaker = max(talk_time_ratio, key=talk_time_ratio.get)
    transcript_segments = [{"speaker": max_speaker, "text": s} for s in sentences]

    questions_per_speaker, total_questions = count_questions(transcript_segments)

    sentiment = analyze_sentiment(full_text)
    insight = extract_insight(full_text)
    roles = identify_roles(talk_time_ratio, questions_per_speaker, transcript_segments)

    # Print results
    print("Talk-time ratio (%):", talk_time_ratio)
    print("Number of questions asked:", total_questions)
    print(f"Longest monologue: Speaker {longest_monologue['speaker']} for {longest_monologue['duration']:.2f} seconds")
    print("Call sentiment:", sentiment)
    print("Actionable insight:", insight)
    print("Identified roles:", roles)

# --- Example usage ---
# Upload your audio file to Colab and set filename here:
audio_filename = "your_sales_call.wav"  # Replace with your file path

# Uncomment below to run analysis (make sure to upload audio file first)
# analyze_call(audio_filename)