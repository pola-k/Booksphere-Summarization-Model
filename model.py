from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import os
import fitz
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

app = Flask(__name__)
CORS(app)

MODEL_DIR = "./pegasus_summarizer"
tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR)

MAX_INPUT_LENGTH = 1024 
SUMMARY_MAX_LENGTH = 128

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def split_text(text, max_tokens=MAX_INPUT_LENGTH):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def generate_summary(text):
    batch = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    gen_out = model.generate(
        **batch,
        max_length=SUMMARY_MAX_LENGTH,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5
    )
    return tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0]

@app.route("/", methods=["GET"])
def home():
    return "📚 Book Summarization API is Running!"

@app.route("/summarize", methods=["POST"])
def summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        return jsonify({"error": "Unsupported file type. Only PDF and TXT allowed."}), 400

    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)
    file.save(temp_path)

    def generate_stream():
        try:
            text = extract_text(temp_path)
            if not text or not text.strip():
                yield "No text found in file.\n"
                return

            chunks = split_text(text)
            total = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                print(f"Currently summarizing chunk number: {idx}/{total}")
                try:
                    summary = generate_summary(chunk)
                    yield f"{summary}\n"
                except Exception as e:
                    yield f"\n\nError summarizing chunk.....\n"
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return Response(stream_with_context(generate_stream()), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)
