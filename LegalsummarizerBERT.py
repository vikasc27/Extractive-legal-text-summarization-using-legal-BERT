# ✅ STEP 1: Install required libraries
!pip install -q transformers pdfplumber gradio sentence-transformers scikit-learn
!pip install -q torch torchvision torchaudio

# ✅ STEP 2: Imports
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import gradio as gr
import re
import numpy as np
from typing import List, Dict, Tuple
import warnings
import os
import tempfile
import time
warnings.filterwarnings('ignore')

# ✅ STEP 3: Load Fine-tuned Legal Model
print("🔄 Loading Your Trained Legal BERT Model...")

model_path = "/content/drive/MyDrive/output_preds"

# Define labels (make sure this matches your training setup)
LABELS = ['FACTS', 'ISSUE', 'ARGUMENT', 'ANALYSIS']

# Set device first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

try:
    # Check if the model directory exists and contains necessary files
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

    if missing_files:
        print(f"⚠️ Warning: Missing files in model directory: {missing_files}")

    # Load tokenizer and model with proper error handling
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        num_labels=len(LABELS)
    )

    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    MODEL_LOADED = True

except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("🔄 Falling back to rule-based classification...")
    MODEL_LOADED = False

# For sentence similarity to avoid repetition
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Similarity model loaded")
except:
    print("⚠️ Warning: Could not load similarity model, will use basic deduplication")
    similarity_model = None

# ✅ STEP 4: Enhanced Sentence Processing for Long Documents

def create_overlapping_chunks(text: str, max_length: int = 400, overlap: int = 50) -> List[str]:
    """Create overlapping text chunks to preserve context in long documents"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())

        if i + max_length >= len(words):
            break

    return chunks

def advanced_sentence_split(text: str) -> List[str]:
    """Enhanced sentence splitting with better handling for legal documents"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)

    # Handle legal citations and abbreviations
    text = re.sub(r'(\bv\.|vs\.|Art\.|Sec\.|para\.|Para\.|cf\.|ibid\.|etc\.|e\.g\.|i\.e\.)',
                  lambda m: m.group(1).replace('.', '<!DOT!>'), text)

    sentences, current_sentence = [], ""

    for line in text.split('\n'):
        line = line.strip().replace('<!DOT!>', '.')
        if not line:
            continue

        # Skip page numbers, headers, footers
        if (len(line.split()) > 2 and not re.match(r'^\d+\.?\s*', line) and
            not line.lower().startswith('page') and
            not re.match(r'^(page \d+|chapter \d+|\d+\s*$)', line.lower())):

            if current_sentence and not current_sentence.endswith(('.', '!', '?')):
                current_sentence += " " + line
            else:
                if current_sentence:
                    sentences.append(current_sentence.strip())
                current_sentence = line

    if current_sentence:
        sentences.append(current_sentence.strip())

    # Enhanced quality filtering for legal documents
    quality_sentences = []
    for sent in sentences:
        words = sent.split()
        if (5 <= len(words) <= 80 and
            len(sent) >= 25 and
            sent.count('.') <= 5 and
            not re.match(r'^\d+\.?\s*', sent) and
            not re.match(r'^(page|chapter|section)\s+\d+', sent.lower())):
            quality_sentences.append(sent)

    return quality_sentences

def classify_with_model_chunked(sentence: str) -> Tuple[str, float]:
    """Enhanced model classification with chunking for long sentences"""
    try:
        # If sentence is too long, split into chunks and aggregate results
        if len(sentence.split()) > 400:
            chunks = create_overlapping_chunks(sentence, max_length=400, overlap=50)
            predictions = []
            confidences = []

            for chunk in chunks:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                    predictions.append(predicted_class)
                    confidences.append(confidence)

            # Aggregate predictions (majority vote with confidence weighting)
            if predictions:
                weighted_votes = defaultdict(float)
                for pred, conf in zip(predictions, confidences):
                    weighted_votes[pred] += conf

                final_prediction = max(weighted_votes, key=weighted_votes.get)
                final_confidence = weighted_votes[final_prediction] / len(predictions)

                return LABELS[final_prediction], final_confidence

        # Standard processing for normal-length sentences
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return LABELS[predicted_class], confidence

    except Exception as e:
        print(f"Error in model classification: {e}")
        return classify_with_enhanced_rules(sentence)

def classify_with_enhanced_rules(sentence: str) -> Tuple[str, float]:
    """Enhanced rule-based classification with confidence scoring"""
    sent_lower = sentence.lower()

    # More comprehensive keyword sets
    keyword_patterns = {
        'FACTS': {
            'primary': ['filed', 'arrested', 'charged', 'occurred', 'incident', 'fir', 'complaint',
                       'case registered', 'investigation', 'evidence', 'witness', 'appellant',
                       'respondent', 'accused', 'victim', 'petitioner', 'prosecution'],
            'secondary': ['crime', 'offence', 'conviction', 'acquittal', 'bail', 'custody',
                         'trial court', 'sessions court', 'magistrate', 'police', 'statement'],
            'legal_entities': ['state of', 'union of india', 'cbi', 'cid', 'police station']
        },
        'ISSUE': {
            'primary': ['whether', 'question', 'issue', 'dispute', 'interpretation', 'jurisdiction',
                       'constitutional', 'legal question', 'point of law', 'matter in dispute'],
            'secondary': ['controversy', 'constitutional validity', 'interpretation of', 'scope of',
                         'applicability of', 'violation of', 'breach of'],
            'legal_refs': ['article', 'section', 'rule', 'regulation', 'provision', 'clause']
        },
        'ARGUMENT': {
            'primary': ['submitted', 'argued', 'contended', 'counsel', 'advocate', 'pleaded',
                       'learned counsel', 'submission', 'argument', 'contention'],
            'secondary': ['plea', 'represented', 'on behalf', 'counsel for', 'argued that',
                         'senior advocate', 'additional solicitor general', 'attorney general'],
            'phrases': ['it is submitted', 'it is argued', 'counsel contends', 'learned counsel']
        },
        'ANALYSIS': {
            'primary': ['held', 'observed', 'court found', 'concluded', 'judgment', 'order',
                       'decision', 'ruling', 'we find', 'it is clear', 'court held'],
            'secondary': ['disposed of', 'dismissed', 'allowed', 'reasoning', 'ratio',
                         'therefore', 'hence', 'accordingly', 'in view of', 'considering'],
            'judicial': ['this court', 'supreme court', 'high court', 'hon\'ble court']
        }
    }

    scores = defaultdict(float)

    # Calculate weighted scores
    for category, patterns in keyword_patterns.items():
        primary_score = sum(3 for keyword in patterns['primary'] if keyword in sent_lower)
        secondary_score = sum(2 for keyword in patterns['secondary'] if keyword in sent_lower)

        # Special pattern matching
        if category == 'ISSUE' and re.search(r'\b(section|article|rule)\s+\d+', sent_lower):
            scores[category] += 4
        elif category == 'FACTS' and ('vs.' in sentence or 'v.' in sentence):
            scores[category] += 4
        elif category == 'ANALYSIS' and re.search(r'\b(para|paragraph)\s+\d+', sent_lower):
            scores[category] += 3

        # Additional scoring for legal phrases
        if 'phrases' in patterns:
            phrase_score = sum(4 for phrase in patterns['phrases'] if phrase in sent_lower)
            scores[category] += phrase_score

        if 'legal_entities' in patterns:
            entity_score = sum(3 for entity in patterns['legal_entities'] if entity in sent_lower)
            scores[category] += entity_score

        if 'legal_refs' in patterns:
            ref_score = sum(2 for ref in patterns['legal_refs'] if ref in sent_lower)
            scores[category] += ref_score

        if 'judicial' in patterns:
            judicial_score = sum(2 for term in patterns['judicial'] if term in sent_lower)
            scores[category] += judicial_score

        total_score = primary_score + secondary_score
        scores[category] += total_score

    # Determine best category and confidence
    if scores:
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        total_possible = sum(scores.values()) if sum(scores.values()) > 0 else 1
        confidence = min(max_score / total_possible, 0.95)
        return best_category, confidence

    return 'FACTS', 0.3

def classify(sentence: str) -> Tuple[str, float]:
    """Enhanced classification with confidence scores"""
    if MODEL_LOADED:
        return classify_with_model_chunked(sentence)
    else:
        return classify_with_enhanced_rules(sentence)

def remove_similar_sentences(sentences: List[str], threshold: float = 0.8) -> List[str]:
    if not sentences or similarity_model is None:
        # Basic deduplication without similarity model
        seen, unique = set(), []
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if sent_clean not in seen and len(sent_clean) > 10:
                unique.append(sent)
                seen.add(sent_clean)
        return unique

    try:
        embeddings = similarity_model.encode(sentences)
        sim_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        sim_matrix = sim_matrix / np.outer(norms, norms)

        unique_indices, used = [], set()
        for i in range(len(sentences)):
            if i in used:
                continue
            unique_indices.append(i)
            for j in range(i + 1, len(sentences)):
                if sim_matrix[i][j] > threshold:
                    used.add(j)
        return [sentences[i] for i in unique_indices]
    except Exception as e:
        print(f"Error in similarity removal: {e}")
        # Fallback to basic deduplication
        seen, unique = set(), []
        for sent in sentences:
            if sent not in seen:
                unique.append(sent)
                seen.add(sent)
        return unique

def smart_summarize(sentences: List[str], max_sentences: int = 5) -> List[str]:
    if not sentences:
        return []

    unique_sentences = remove_similar_sentences(sentences, threshold=0.75)
    scored = []

    for sent in unique_sentences:
        score = 0
        words = sent.split()

        # Length scoring
        if 10 <= len(words) <= 40:
            score += 3
        elif 5 <= len(words) <= 60:
            score += 1

        # Legal terms scoring
        legal_terms = [
            'court', 'appellant', 'respondent', 'petitioner', 'article',
            'section', 'act', 'constitution', 'law', 'judgment', 'order',
            'held', 'observed', 'contended', 'argued'
        ]
        score += sum(1 for term in legal_terms if term in sent.lower())

        # Sentence completion
        if sent.endswith(('.', '?', '!')):
            score += 1

        # Penalty for very short sentences
        if len(words) < 5:
            score -= 5

        scored.append((sent, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    final, seen = [], []
    for sent, score in scored:
        if len(final) >= max_sentences:
            break

        # Check for content overlap
        if all(len(set(sent.lower().split()) & set(s.lower().split())) /
               len(set(sent.lower().split()) | set(s.lower().split())) < 0.6
               for s in seen):
            final.append(sent)
            seen.append(sent)

    return final

def extract_fallback_content(full_text: str, all_sentences: List[str]) -> Dict[str, List[str]]:
    fallback = {'FACTS': [], 'ISSUE': [], 'ARGUMENT': [], 'ANALYSIS': []}

    # Case name extraction
    case_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)\s+v(?:s)?\.?\s+([A-Z][a-z]+ [A-Z][a-z]+)', full_text)
    if case_match:
        fallback['FACTS'].append(f"Case involves {case_match.group(1)} versus {case_match.group(2)}.")

    # Court identification
    court_match = re.search(r'(Supreme Court|High Court|Sessions Court|District Court)', full_text, re.I)
    if court_match:
        fallback['FACTS'].append(f"Proceedings before the {court_match.group(1)}.")

    # Date extraction
    date_match = re.search(r'(?:dated?|on|judgment)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', full_text)
    if date_match:
        fallback['FACTS'].append(f"Judgment dated {date_match.group(1)}.")

    # Constitutional/statutory matters
    if re.search(r'article|section|constitution', full_text, re.I):
        fallback['ISSUE'].append("Constitutional or statutory questions involved.")

    # Party arguments
    if re.search(r'appellant|petitioner', full_text, re.I):
        fallback['ARGUMENT'].append("Appellant presented arguments against lower court's decision.")

    if re.search(r'respondent|state', full_text, re.I):
        fallback['ARGUMENT'].append("Respondent defended the decision.")

    # Court's analysis
    if re.search(r'judgment|held|disposed|concluded', full_text, re.I):
        fallback['ANALYSIS'].append("Court provided reasoning and final judgment.")

    return fallback

# ✅ STEP 5: Simplified Main Function - CLEAN OUTPUT ONLY
def process_legal_document(pdf_file) -> str:
    try:
        # Handle the uploaded file
        if pdf_file is None:
            return "❌ Please upload a PDF file"

        # Gradio passes the file path as a string
        if isinstance(pdf_file, str):
            pdf_path = pdf_file
        else:
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n".join([p.extract_text() or '' for p in pdf.pages])
        except Exception as pdf_error:
            return f"❌ Error reading PDF: {str(pdf_error)}"

        if not full_text.strip():
            return "❌ No readable text found in PDF"

        # Enhanced sentence processing
        sentences = advanced_sentence_split(full_text)

        # Classify sentences
        classified = defaultdict(list)

        for sentence in sentences:
            prediction, confidence = classify(sentence)
            classified[prediction].append(sentence)

        # Fallback content extraction
        fallback = extract_fallback_content(full_text, sentences)
        summary = {}

        # Generate summary for each category
        for cat in LABELS:
            if classified[cat]:
                points = smart_summarize(classified[cat], max_sentences=5)
                if len(points) < 3:
                    extra = [s for s in classified[cat] if s not in points][:5 - len(points)]
                    points += extra
                summary[cat] = points[:5]
            else:
                summary[cat] = fallback[cat] or [f"No explicit {cat.lower()} content found."]

        # Format CLEAN output - ONLY the four categories
        output = "🏛️ LEGAL DOCUMENT SUMMARY\n" + "="*50 + "\n\n"
        icons = {'FACTS': '📋', 'ISSUE': '❓', 'ARGUMENT': '⚖️', 'ANALYSIS': '🔍'}

        # Summary sections ONLY
        for cat in LABELS:
            output += f"{icons[cat]} {cat}\n" + "-"*30 + "\n"
            for i, point in enumerate(summary[cat], 1):
                output += f"{i}. {point}\n\n"
            output += "\n"

        return output

    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

# ✅ STEP 6: Simplified Gradio Interface
def create_gradio_interface():
    with gr.Blocks(title="Legal Document Summarizer", theme=gr.themes.Soft()) as demo:
        # Header with gradient
        gr.HTML("""
            <div style="text-align: center; padding: 25px;
                        background: linear-gradient(90deg, #1e3c72, #2a5298);
                        color: white; border-radius: 15px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 38px;">⚖️ Legal Document Summarizer</h1>
                <p style="font-size: 18px; margin-top: 10px;">
                    Upload a PDF and get a structured summary (Facts • Issues • Arguments • Analysis)
                </p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # File upload card
                gr.HTML("""
                    <div style="padding: 15px; border-radius: 12px;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                                background: #f9fafb; margin-bottom: 15px;">
                        <h3 style="margin: 0; color: #1e3c72;">📂 Upload Document</h3>
                        <p style="font-size: 14px; color: #555;">Supported format: PDF</p>
                    </div>
                """)
                pdf_input = gr.File(
                    label="Upload Legal Document",
                    file_types=[".pdf"],
                    file_count="single"
                )

                # Process button styled
                process_btn = gr.Button(
                    "🔍 Analyze Document",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                # Output - card style
                gr.HTML("""
                    <div style="padding: 15px; border-radius: 12px;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
                                background: #ffffff; margin-bottom: 10px;">
                        <h3 style="margin: 0; color: #2a5298;">📋 Document Summary</h3>
                        <p style="font-size: 14px; color: #666;">Your structured summary will appear below</p>
                    </div>
                """)
                output_text = gr.Textbox(
                    label="Summary",
                    lines=25,
                    max_lines=30,
                    placeholder="Upload a PDF document to see the summary here...",
                )

        # Event handlers
        process_btn.click(
            fn=process_legal_document,
            inputs=[pdf_input],
            outputs=[output_text],
            show_progress=True
        )

        # Auto-process when file is uploaded
        pdf_input.change(
            fn=process_legal_document,
            inputs=[pdf_input],
            outputs=[output_text],
            show_progress=True
        )

    return demo


# ✅ STEP 7: Launch the Interface
print("🚀 Initializing Legal Summarizer...")
print(f"📊 Model Status: {'✅ Custom model loaded' if MODEL_LOADED else '⚠️ Using rule-based classification'}")

# Create and launch the interface
demo = create_gradio_interface()

# Launch
demo.launch(
    share=True,
    debug=True,
    show_error=True,
    server_name="0.0.0.0",
    server_port=7860
)