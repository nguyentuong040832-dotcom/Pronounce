import whisper
import librosa
import numpy as np
import re
import textdistance
import difflib
from jiwer import wer, cer
from datetime import datetime
import warnings
from typing import Dict, List, Any

warnings.filterwarnings("ignore")


class PronunciationResult:
    def __init__(self, transcription: str, confidence: float, scores: Dict[str, Any], feedback: Dict[str, Any],
                 language: str, processing_time: float):
        self.transcription = transcription
        self.confidence = confidence
        self.scores = scores
        self.feedback = feedback
        self.language = language
        self.processing_time = processing_time
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "transcription": self.transcription,
            "confidence": self.confidence,
            "scores": self.scores,
            "feedback": self.feedback,
            "language": self.language,
            "processing_time": round(self.processing_time, 2),
            "timestamp": self.timestamp
        }


class PronunciationScorer:
    def __init__(self, model_size="base", device="auto"):
        # Choose device
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Loading Whisper model '{model_size}' on device '{self.device}'...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        print("Model loaded.")

    def preprocess_audio(self, audio_path: str, sr_target=16000, max_duration=30.0):
        audio, sr = librosa.load(audio_path, sr=sr_target, mono=True)
        duration = len(audio) / sr
        if duration < 0.3:
            raise ValueError("Audio too short (<0.3s)")
        if duration > max_duration:
            audio = audio[: int(max_duration * sr)]
        audio, _ = librosa.effects.trim(audio, top_db=15)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        return audio, sr

    def transcribe_audio(self, audio_path: str, language='en'):
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            fp16=(self.whisper_model.device.type == "cuda"),
            verbose=False
        )
        avg_conf = 0.0
        wc = 0
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "probability" in w:
                    try:
                        avg_conf += float(w["probability"])
                        wc += 1
                    except:
                        pass
        avg_conf = (avg_conf / wc) if wc > 0 else 0.0
        return result.get("text", "").strip(), round(avg_conf, 3), result.get("segments", [])

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"
        }
        for k, v in contractions.items():
            text = text.replace(k, v)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def calculate_scores(self, target_text: str, transcribed_text: str) -> Dict[str, Any]:
        target_clean = self.clean_text(target_text)
        transcribed_clean = self.clean_text(transcribed_text)

        if not target_clean or not transcribed_clean:
            if target_clean == transcribed_clean:
                return self._perfect_score(target_clean)
            else:
                return self._zero_score(target_clean, transcribed_clean)

        try:
            word_error_rate = wer(target_clean, transcribed_clean)
            char_error_rate = cer(target_clean, transcribed_clean)
        except:
            word_error_rate = 1.0
            char_error_rate = 1.0

        target_words = target_clean.split()
        transcribed_words = transcribed_clean.split()
        jaccard_score = (len(set(target_words) & set(transcribed_words)) / len(
            set(target_words) | set(transcribed_words))) if (target_words or transcribed_words) else 1.0
        levenshtein_score = textdistance.levenshtein.normalized_similarity(target_clean, transcribed_clean)

        word_accuracy = max(0, (1 - word_error_rate) * 100)
        char_accuracy = max(0, (1 - char_error_rate) * 100)

        # ✅ Nếu chỉ có 1 từ -> tính điểm "mềm" hơn
        if len(target_words) == 1:
            overall_score = (levenshtein_score * 0.6 + (char_accuracy / 100) * 0.4) * 100
        else:
            overall_score = (
                    word_accuracy * 0.4 +
                    char_accuracy * 0.3 +
                    jaccard_score * 100 * 0.2 +
                    levenshtein_score * 100 * 0.1
            )

        return {
            "word_error_rate": round(word_error_rate, 4),
            "character_error_rate": round(char_error_rate, 4),
            "word_accuracy": round(word_accuracy, 2),
            "character_accuracy": round(char_accuracy, 2),
            "jaccard_similarity": round(jaccard_score * 100, 2),
            "levenshtein_similarity": round(levenshtein_score * 100, 2),
            "overall_score": round(overall_score, 2),
            "target_text": target_clean,
            "transcribed_text": transcribed_clean
        }

    def _perfect_score(self, target_clean: str) -> Dict[str, Any]:
        return {
            "word_error_rate": 0.0,
            "character_error_rate": 0.0,
            "word_accuracy": 100.0,
            "character_accuracy": 100.0,
            "jaccard_similarity": 100.0,
            "levenshtein_similarity": 100.0,
            "overall_score": 100.0,
            "target_text": target_clean,
            "transcribed_text": target_clean
        }

    def _zero_score(self, target_clean: str, transcribed_clean: str) -> Dict[str, Any]:
        return {
            "word_error_rate": 1.0,
            "character_error_rate": 1.0,
            "word_accuracy": 0.0,
            "character_accuracy": 0.0,
            "jaccard_similarity": 0.0,
            "levenshtein_similarity": 0.0,
            "overall_score": 0.0,
            "target_text": target_clean,
            "transcribed_text": transcribed_clean
        }

    # ---------- Feedback ----------
    def get_detailed_feedback(self, target_text: str, transcribed_text: str, confidence: float) -> Dict[str, Any]:
        target_words = self.clean_text(target_text).split()
        transcribed_words = self.clean_text(transcribed_text).split()

        feedback = {
            "correct_words": [],
            "incorrect_words": [],
            "missing_words": [],
            "extra_words": [],
            "suggestions": []
        }

        matcher = difflib.SequenceMatcher(None, target_words, transcribed_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                feedback["correct_words"].extend(target_words[i1:i2])
            elif tag == 'replace':
                for idx in range(max(i2 - i1, j2 - j1)):
                    tw = target_words[i1 + idx] if i1 + idx < i2 else None
                    pw = transcribed_words[j1 + idx] if j1 + idx < j2 else None
                    if tw and pw:
                        sim = textdistance.levenshtein.normalized_similarity(tw, pw)
                        feedback["incorrect_words"].append({
                            "target": tw,
                            "pronounced": pw,
                            "similarity": round(sim, 3),
                            "score": int(round(sim * 100))
                        })
                    elif tw:
                        feedback["missing_words"].append(tw)
                    elif pw:
                        feedback["extra_words"].append(pw)
            elif tag == 'delete':
                feedback["missing_words"].extend(target_words[i1:i2])
            elif tag == 'insert':
                feedback["extra_words"].extend(transcribed_words[j1:j2])

        feedback["suggestions"] = self.generate_suggestions(feedback, confidence)
        return feedback

    def generate_suggestions(self, feedback: Dict[str, Any], confidence: float) -> List[str]:
        suggestions = []
        if confidence < 0.6:
            suggestions.append("Hãy nói to và rõ ràng hơn")
        elif confidence < 0.8:
            suggestions.append("Hãy cải thiện độ rõ ràng")

        if feedback["missing_words"]:
            cnt = len(feedback["missing_words"])
            if cnt == 1:
                suggestions.append(f"Bạn đã bỏ sót từ: '{feedback['missing_words'][0]}'")
            else:
                suggestions.append(f"Bạn đã bỏ sót {cnt} từ: {', '.join(feedback['missing_words'][:3])}")

        if feedback["extra_words"]:
            suggestions.append(f"Bạn đã thêm {len(feedback['extra_words'])} từ không cần thiết")

        if feedback["incorrect_words"]:
            for item in feedback["incorrect_words"][:3]:
                if item["similarity"] > 0.6:
                    suggestions.append(
                        f"'{item['target']}' → bạn nói '{item['pronounced']}' (similarity {item['similarity']})")
                else:
                    suggestions.append(f"Từ '{item['target']}' cần được phát âm lại")

        if not suggestions:
            suggestions.append("Xuất sắc! Phát âm rất chính xác!")

        return suggestions

    # ---------- Main entry ----------
    def score_pronunciation(self, audio_path: str, target_text: str, language: str = "en") -> PronunciationResult:
        start = datetime.now()
        # optional pre-processing if needed
        # audio, sr = self.preprocess_audio(audio_path)  # not used: whisper transcribe handles file paths fine
        transcription, confidence, segments = self.transcribe_audio(audio_path, language=language)
        scores = self.calculate_scores(target_text, transcription)
        feedback = self.get_detailed_feedback(target_text, transcription, confidence)
        processing_time = (datetime.now() - start).total_seconds()
        return PronunciationResult(transcription, confidence, scores, feedback, language, processing_time)
