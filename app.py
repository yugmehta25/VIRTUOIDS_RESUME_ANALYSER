from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from utils.extracter import extract_text_from_pdf, extract_skills
from utils.matcher import calculate_match_score

app = Flask(__name__, static_folder='../front_end', static_url_path='/')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing resume or job description'}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)

    resume_text = extract_text_from_pdf(resume_path)
    skills = extract_skills(resume_text)
    match_score = calculate_match_score(resume_text, job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_description, resume_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return jsonify({
        'skills': skills,
        'match_score': round(score * 100, 2),
        'message': 'Analysis complete!'
    })

if __name__ == '__main__':
    app.run(debug=True)
