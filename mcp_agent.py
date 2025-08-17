import argparse
import io
import os
import re
import sys
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ---- Env & deps ----
from dotenv import load_dotenv
load_dotenv()

# Google API
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth.exceptions
import pickle

# Parsing
from docx import Document as DocxDocument
from pypdf import PdfReader

# NLP
import spacy
from rapidfuzz import fuzz, process

# Templating
from jinja2 import Template

# Optional LLM
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------------------------
# Config & Constants
# ---------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive"]

DEFAULT_OUTPUT_FOLDER = os.getenv("MCP_OUTPUT_FOLDER", "MCP Agent Outputs")
USE_LLM = os.getenv("MCP_USE_LLM", "false").lower() == "true"
OPENAI_MODEL = os.getenv("MCP_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SUPPORTED_EXPORTS = {
    # Google Docs export to text
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/plain",
}

TEXTY_MIMES = {
    "text/plain",
    "application/json",
    "application/xml",
    "text/markdown",
}

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MIME = "application/pdf"

# ---------------------------------------
# Helpers
# ---------------------------------------

def google_auth():
    """
    Returns an authorized Drive service using OAuth client credentials.
    Creates/refreshes token.pickle in the working directory.
    """
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except google.auth.exceptions.RefreshError:
                creds = None
        if not creds:
            if not os.path.exists("credentials.json"):
                print("Missing credentials.json. Download your OAuth client secrets and place them here.", file=sys.stderr)
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("drive", "v3", credentials=creds)


def get_file_metadata(service, file_id: str) -> dict:
    return service.files().get(fileId=file_id, fields="id,name,mimeType,parents").execute()


def download_file_as_text(service, file_id: str, mime_type: str) -> str:
    """
    For Google-native files, export as text/plain.
    For others, download bytes, then parse.
    """
    meta = get_file_metadata(service, file_id)
    name = meta["name"]
    mt = meta["mimeType"]

    if mt in SUPPORTED_EXPORTS:
        request = service.files().export_media(fileId=file_id, mimeType=SUPPORTED_EXPORTS[mt])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        text = fh.getvalue().decode("utf-8", errors="ignore")
        return text

    # Otherwise, download the raw file content
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    content = fh.getvalue()

    # Route by mime
    if mt in TEXTY_MIMES:
        return content.decode("utf-8", errors="ignore")
    elif mt == DOCX_MIME or name.lower().endswith(".docx"):
        return parse_docx_bytes(content)
    elif mt == PDF_MIME or name.lower().endswith(".pdf"):
        return parse_pdf_bytes(content)
    else:
        # best-effort fallback
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def parse_docx_bytes(content: bytes) -> str:
    bio = io.BytesIO(content)
    doc = DocxDocument(bio)
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    return "\n".join(parts)


def parse_pdf_bytes(content: bytes) -> str:
    bio = io.BytesIO(content)
    reader = PdfReader(bio)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


# ---------------------------------------
# NLP / Analysis
# ---------------------------------------

@dataclass
class Section:
    title: str
    content: str
    score: float


def basic_section_split(text: str) -> List[Section]:
    """
    Heuristic: split by heading-like lines and big gaps.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    sections: List[Section] = []
    current_title = "General"
    current_buf = []

    def flush():
        nonlocal sections, current_title, current_buf
        if current_buf:
            content = "\n".join(current_buf).strip()
            if content:
                sections.append(Section(title=current_title, content=content, score=0.0))
            current_buf = []

    heading_re = re.compile(r"^\s*([A-Z][A-Za-z0-9\s&\-/]{2,})\s*$")

    for line in lines:
        if heading_re.match(line) and len(line.split()) <= 8:
            # probable heading
            flush()
            current_title = line.strip()
        else:
            current_buf.append(line)

    flush()
    return sections


def score_sections_by_skill(sections: List[Section], skill: str, nlp) -> List[Section]:
    """
    Score by fuzzy match + token overlap with synonyms.
    """
    synonyms = generate_skill_synonyms(skill, nlp)
    key_phrases = [skill] + synonyms

    def score_text(t: str) -> float:
        # fuzzy ratio with each key phrase + simple frequency
        fuzz_max = max([fuzz.token_set_ratio(t, k) for k in key_phrases]) if key_phrases else 0
        freq = sum(t.lower().count(k.lower()) for k in key_phrases)
        return 0.7 * (fuzz_max / 100.0) + 0.3 * min(freq / 10.0, 1.0)

    for s in sections:
        s.score = 0.4 * score_text(s.title) + 0.6 * score_text(s.content[:1200])
    sections.sort(key=lambda s: s.score, reverse=True)
    return sections


def generate_skill_synonyms(skill: str, nlp) -> List[str]:
    """
    Very light synonym generator:
    - spaCy lemmas
    - handwave expansions for common domains
    """
    expansions = {
        "python": ["py", "pandas", "numpy", "scikit-learn", "flask", "django", "fastapi", "pytest"],
        "machine learning": ["ml", "scikit-learn", "classification", "regression", "xgboost", "lightgbm"],
        "project management": ["agile", "scrum", "kanban", "stakeholders", "risk management", "pmp"],
        "data engineering": ["spark", "airflow", "etl", "kafka", "hadoop", "hive"],
    }

    toks = [t.lemma_.lower() for t in nlp(skill)]
    base = set(toks + [skill.lower()])
    extra = []
    for k, v in expansions.items():
        if any(term in " ".join(base) for term in [k] + k.split()):
            extra += v
    return list(set(extra))


# ---------------------------------------
# Generation (Templates)
# ---------------------------------------

CV_TEMPLATE = Template("""\
<h1>{{ name }}</h1>
<p><strong>Role:</strong> {{ target_role }} | <strong>Focus:</strong> {{ skill }}<br>
<strong>Email:</strong> {{ email }} | <strong>Phone:</strong> {{ phone }} | <strong>Location:</strong> {{ location }}</p>

<h2>Professional Summary</h2>
<p>{{ summary }}</p>

<h2>Core {{ skill }} Skills</h2>
<ul>
{% for s in top_skills %}
<li>{{ s }}</li>
{% endfor %}
</ul>

<h2>Experience</h2>
{% for exp in experiences %}
<h3>{{ exp.title }} — {{ exp.company }} ({{ exp.start }} – {{ exp.end }})</h3>
<ul>
{% for b in exp.bullets %}
<li>{{ b }}</li>
{% endfor %}
</ul>
{% endfor %}

<h2>Projects</h2>
{% for pr in projects %}
<h3>{{ pr.name }}</h3>
<p>{{ pr.desc }}</p>
<ul>
{% for b in pr.bullets %}
<li>{{ b }}</li>
{% endfor %}
</ul>
{% endfor %}

<h2>Certifications</h2>
<ul>
{% for c in certs %}
<li>{{ c }}</li>
{% endfor %}
</ul>

<h2>Education</h2>
<ul>
{% for e in education %}
<li>{{ e }}</li>
{% endfor %}
</ul>
""")

TOC_TEMPLATE = Template("""\
<h1>{{ skill }} — Table of Contents</h1>
<ol>
{% for item in items %}
<li><strong>{{ item.title }}</strong>{% if item.note %}: {{ item.note }}{% endif %}</li>
{% endfor %}
</ol>
""")

# ---------------------------------------
# Optional LLM Refinement
# ---------------------------------------

def maybe_refine_with_llm(title: str, html: str, skill: str, which: str) -> str:
    if not USE_LLM:
        return html
    if not (OPENAI_API_KEY and OpenAI):
        print("LLM disabled (missing key or SDK). Returning template output.", file=sys.stderr)
        return html
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
You are an expert CV editor. Improve the following {which} HTML so it's concise, impact-focused, and tailored to the skill "{skill}".
Preserve HTML structure but fix phrasing and make bullets achievement-oriented with measurable impact where appropriate.

HTML:
{html}
"""
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role":"system","content":"You rewrite HTML content for professional documents without breaking tags."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_output_tokens=2000
        )
        # Extract text output
        out = resp.output_text if hasattr(resp, "output_text") else None
        return out or html
    except Exception as e:
        print(f"LLM refinement failed: {e}", file=sys.stderr)
        return html


# ---------------------------------------
# Drive Uploads
# ---------------------------------------

def find_or_create_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    q = f"mimeType='application/vnd.google-apps.folder' and name='{name.replace(\"'\",\"\\'\")}' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = service.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_html_as_gdoc(service, folder_id: str, name: str, html: str) -> str:
    """
    Convert HTML -> Google Doc by uploading as google-apps.document with multipart.
    """
    media_body = MediaIoBaseUpload(io.BytesIO(html.encode("utf-8")), mimetype="text/html", resumable=True)
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }
    created = service.files().create(body=file_metadata, media_body=media_body, fields="id").execute()
    return created["id"]


# ---------------------------------------
# Extraction from sections (simple heuristics)
# ---------------------------------------

def extract_candidate_bullets(text: str) -> List[str]:
    # Grab lines that look like bullets or accomplishments
    lines = [l.strip(" •-\t") for l in text.splitlines()]
    return [l.strip() for l in lines if len(l.strip()) > 0 and (l.strip().startswith(("*","-","•")) or len(l.split()) >= 5)]

def synthesize_experiences(sections: List[Section]) -> List[Dict]:
    exps = []
    for s in sections:
        if any(k in s.title.lower() for k in ["experience","employment","work","career"]):
            bullets = extract_candidate_bullets(s.content)[:8]
            exps.append({
                "title": "Title (from doc)" if not bullets else (bullets[0][:50] + "..." if len(bullets[0])>50 else bullets[0]),
                "company": "Company",
                "start": "—",
                "end": "Present",
                "bullets": bullets[:6] if bullets else []
            })
    # If nothing labeled, fallback to top relevant as pseudo-experience
    if not exps and sections:
        s = sections[0]
        bullets = extract_candidate_bullets(s.content)[:6]
        exps.append({
            "title": s.title,
            "company": "—",
            "start": "—",
            "end": "—",
            "bullets": bullets
        })
    return exps[:4]

def synthesize_projects(sections: List[Section]) -> List[Dict]:
    projects = []
    for s in sections:
        if any(k in s.title.lower() for k in ["project","portfolio","case study","case-study"]):
            bullets = extract_candidate_bullets(s.content)[:6]
            projects.append({
                "name": s.title,
                "desc": (s.content.split("\n")[0] if s.content else "")[:180],
                "bullets": bullets[:5]
            })
    # fallback: pick next most relevant as projects
    if not projects:
        for s in sections[1:3]:
            bullets = extract_candidate_bullets(s.content)[:4]
            projects.append({"name": s.title, "desc": s.content[:160], "bullets": bullets})
    return projects[:3]

def synthesize_skills(text: str, skill: str, nlp) -> List[str]:
    tokens = [t.text for t in nlp(text) if t.is_alpha]
    # crude frequency + keep top distinct items around the skill domain
    counts: Dict[str,int] = {}
    for t in tokens:
        low = t.lower()
        if len(low) <= 2: continue
        counts[low] = counts.get(low,0)+1
    # prefer items containing or near the main skill terms
    keys = list(counts.items())
    keys.sort(key=lambda kv: kv[1], reverse=True)
    top = []
    for k, _ in keys:
        if len(top) >= 12: break
        if k in {"with","from","using","into","that","this","have","will","role","team","data","work","tools"}:
            continue
        if fuzz.partial_ratio(k, skill.lower()) >= 60 or counts[k] >= 3:
            top.append(k.capitalize())
    # seed main skill if missing
    if skill.capitalize() not in top:
        top = [skill.capitalize()] + top
    return top[:12]

def synthesize_summary(sections: List[Section], skill: str) -> str:
    # Make a short summary based on top section
    if not sections:
        return f"{skill}-focused professional with hands-on experience; adaptable, outcome-driven."
    s = sections[0]
    first_line = (s.content.strip().split("\n") or [""])[0][:220]
    return f"{skill}-focused professional with experience in {s.title.lower()}. {first_line}"

def synthesize_certs(text: str) -> List[str]:
    pat = re.compile(r"(certified|certificate|certification)\s+([A-Za-z0-9\-\s\+]{2,})", re.I)
    certs = []
    for m in pat.finditer(text):
        certs.append(m.group(0).strip().rstrip("."))
    return list(dict.fromkeys(certs))[:6]

def synthesize_education(text: str) -> List[str]:
    uni = re.compile(r"(B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|B\.?E|M\.?E|MBA|Ph\.?D|Bachelor|Master).{0,60}(University|College|Institute)", re.I)
    edus = []
    for m in uni.finditer(text):
        edus.append(m.group(0).strip().rstrip("."))
    return list(dict.fromkeys(edus))[:4]

# ---------------------------------------
# Orchestrator
# ---------------------------------------

def run_pipeline(skill: str, source_file_id: Optional[str], output_parent: Optional[str]):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Authenticating with Google...")
    service = google_auth()

    nlp = spacy.load("en_core_web_sm")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Reading source...")
    if source_file_id:
        meta = get_file_metadata(service, source_file_id)
        raw_text = download_file_as_text(service, source_file_id, meta["mimeType"])
        parent_hint = (meta.get("parents") or [None])[0]
    else:
        raise SystemExit("Please provide --source_file_id")

    print(f"Source length: {len(raw_text)} chars")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Analyzing content...")
    sections = basic_section_split(raw_text)
    sections = score_sections_by_skill(sections, skill, nlp)

    # Synthesize fields
    text_all = raw_text
    experiences = synthesize_experiences(sections)
    projects = synthesize_projects(sections)
    top_skills = synthesize_skills(text_all, skill, nlp)
    summary = synthesize_summary(sections, skill)
    certs = synthesize_certs(text_all)
    education = synthesize_education(text_all)

    # minimal contact block (could be extracted if present)
    name = "Your Name"
    email = "email@example.com"
    phone = "+91-XXXXXXXXXX"
    location = "City, Country"
    target_role = f"{skill} Specialist"

    cv_html = CV_TEMPLATE.render(
        name=name,
        email=email,
        phone=phone,
        location=location,
        target_role=target_role,
        skill=skill,
        summary=summary,
        top_skills=top_skills,
        experiences=experiences,
        projects=projects,
        certs=certs if certs else ["—"],
        education=education if education else ["—"],
    )

    toc_items = [
        {"title": "Professional Summary", "note": f"Tailored to {skill}"},
        {"title": f"Core {skill} Skills", "note": "Keywords & tools"},
        {"title": "Experience", "note": "Relevant roles and impact"},
        {"title": "Projects", "note": "Hands-on work and outcomes"},
        {"title": "Certifications", "note": "Relevant credentials"},
        {"title": "Education", "note": "Degrees & institutions"},
    ]
    toc_html = TOC_TEMPLATE.render(skill=skill, items=toc_items)

    # Optional LLM polishing
    print(f"[{datetime.now().isoformat(timespec='seconds')}] (Optional) Refining with LLM = {USE_LLM}")
    cv_html = maybe_refine_with_llm(f"{skill} CV", cv_html, skill, which="CV")
    toc_html = maybe_refine_with_llm(f"{skill} TOC", toc_html, skill, which="TOC")

    # Output folder
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Uploading outputs to Drive...")
    parent = output_parent or parent_hint
    folder_id = find_or_create_folder(service, DEFAULT_OUTPUT_FOLDER, parent_id=parent)

    cv_id = upload_html_as_gdoc(service, folder_id, f"{skill}_CV", cv_html)
    toc_id = upload_html_as_gdoc(service, folder_id, f"{skill}_TOC", toc_html)

    print(f"Created CV Doc ID: {cv_id}")
    print(f"Created TOC Doc ID: {toc_id}")
    print("Done.")


# ---------------------------------------
# CLI
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Agent: Generate skill-focused CV & TOC from a source document in Google Drive.")
    parser.add_argument("--skill", required=True, help="Skill/subject to focus on (e.g., Python, Machine Learning)")
    parser.add_argument("--source_file_id", required=True, help="Google Drive file ID of the source document")
    parser.add_argument("--output_parent", default=None, help="Optional parent folder ID under which to create the output folder")
    args = parser.parse_args()

    run_pipeline(args.skill, args.source_file_id, args.output_parent)


if __name__ == "__main__":
    main()
