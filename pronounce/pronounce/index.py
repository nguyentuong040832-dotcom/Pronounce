import os
import time
import traceback
import tempfile
import requests
from flask import Flask, request, jsonify
from scorer import PronunciationScorer

MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
DEVICE = os.environ.get("INFERENCE_DEVICE", "auto")

app = Flask(__name__)

# Load model once at startup
scorer = PronunciationScorer(model_size=MODEL_SIZE, device=DEVICE)

@app.route("/score-pronunciation", methods=["POST"])
def score_pronunciation():
    try:
        target_text = request.form.get("target_text")
        language = request.form.get("language", "en")
        file = request.files.get("audio_file")
        file_url = request.form.get("audio_file")

        if not target_text:
            return jsonify({"error": "target_text is required"}), 400

        tmp_path = None
        # Nếu có file upload -> lưu tạm
        if file:
            suffix = os.path.splitext(file.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp)
                tmp_path = tmp.name

        # Nếu không có file upload nhưng có URL -> tải về
        elif file_url:
            suffix = ".mp3" if file_url.endswith(".mp3") else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                r = requests.get(file_url, stream=True)
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
        else:
            return jsonify({"error": "audio_file (file or URL) is required"}), 400

        start_time = time.time()
        result = scorer.score_pronunciation(tmp_path, target_text, language=language)

        # dọn file tạm
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return jsonify(result.to_dict()), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
