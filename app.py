from flask import Flask, render_template, request
import whisper
import os
from datetime import datetime

app = Flask(__name__)
model = whisper.load_model("tiny")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio = request.files["audio"]
        if audio:
            filename = datetime.now().strftime("%Y%m%d%H%M%S_") + audio.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            audio.save(filepath)

            result = model.transcribe(filepath, language="es", verbose=True)
            segments = result.get("segments", [])
            return render_template("result.html", segments=segments, filename=audio.filename)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
