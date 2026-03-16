import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import spacy
import tiktoken
from pypdf import PdfReader

from config.settings import CHUNK_SIZE
from monitoring.logger import get_logger

logger = get_logger(__name__)

# load NLP model once
nlp = spacy.load("en_core_web_sm")

# tokenizer for embeddings
enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class DocumentChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_name: str = ""
    text: str = ""
    token_count: int = 0
    chunk_index: int = 0


# -------------------------------
# TEXT EXTRACTION
# -------------------------------

def _extract_text_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text(path: Path) -> str:

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_text_pdf(path)

    if suffix in (".txt", ".md"):
        return _extract_text_txt(path)

    raise ValueError(f"Unsupported file type: {suffix}")


# -------------------------------
# TEXT CLEANING
# -------------------------------

def clean_text(text: str) -> str:

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    return text.strip()


# -------------------------------
# SECTION SPLITTING
# -------------------------------

def split_sections(text: str) -> List[str]:
    """
    Split research papers into sections like:
    Introduction, Methods, Conclusion etc.
    """

    sections = re.split(r"\n[A-Z][A-Za-z ]{3,}\n", text)

    if len(sections) < 2:
        return [text]

    return sections


# -------------------------------
# SENTENCE CHUNKING
# -------------------------------
# -------------------------------
# SENTENCE CHUNKING (with overlap)
# -------------------------------

def build_chunks(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Build overlapping chunks using sentence boundaries.

    chunk_size: maximum tokens per chunk
    chunk_overlap: overlapping tokens between chunks
    """

    doc = nlp(text)

    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:

        tokens = enc.encode(sentence)
        token_len = len(tokens)

        # If adding this sentence exceeds chunk size → finalize chunk
        if current_tokens + token_len > chunk_size and current_chunk:

            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Create overlap
            overlap_sentences = []
            overlap_tokens = 0

            for s in reversed(current_chunk):

                s_tokens = len(enc.encode(s))

                if overlap_tokens + s_tokens > chunk_overlap:
                    break

                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_chunk = overlap_sentences
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------------------
# MAIN DOCUMENT PIPELINE
# -------------------------------

def process_document(file_path: Path) -> List[DocumentChunk]:

    logger.info(f"Processing document: {file_path.name}")

    raw_text = extract_text(file_path)

    cleaned = clean_text(raw_text)

    if not cleaned:
        logger.warning(f"No text extracted from {file_path.name}")
        return []

    sections = split_sections(cleaned)

    all_chunks: List[str] = []

    for section in sections:
        section_chunks = build_chunks(section)
        all_chunks.extend(section_chunks)

    chunks: List[DocumentChunk] = []

    for idx, chunk_text in enumerate(all_chunks):

        chunk = DocumentChunk(
            document_name=file_path.name,
            text=chunk_text,
            token_count=len(enc.encode(chunk_text)),
            chunk_index=idx,
        )

        chunks.append(chunk)

    logger.info(f"{len(chunks)} chunks produced from {file_path.name}")

    return chunks