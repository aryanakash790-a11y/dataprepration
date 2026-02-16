"""
text_cleaner.py

Importable module for text extraction & cleaning from various document formats.
Key API:
 - get_spark_session(app_name="TextCleaning", master="local[*]") -> SparkSession
 - stop_spark_session(spark: SparkSession)
 - extract_and_clean(file_path: str, output_path: Optional[str]=None,
                     remove_pii: bool = True, spark: Optional[SparkSession] = None) -> str

Notes:
 - Heavy libraries (PyMuPDF/pdfplumber, python-docx, bs4, striprtf, odfpy) are imported lazily inside
   specific extractors to avoid import-time failures.
 - If a SparkSession is not provided to extract_and_clean, one will be created and stopped automatically.
"""

from pathlib import Path
import re
import unicodedata
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------
# Spark helpers (lazy import inside functions)
# ---------------------------------------
def get_spark_session(app_name: str = "TextCleaning", master: str = "local[*]"):
    """
    Create and return a SparkSession. Lazily imports pyspark.
    """
    try:
        from pyspark.sql import SparkSession
    except Exception as e:
        raise ImportError("PySpark is required to create SparkSession. Install pyspark.") from e

    spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
    return spark


def stop_spark_session(spark):
    """Stop the provided SparkSession (if not None)."""
    try:
        spark.stop()
    except Exception:
        logger.exception("Error stopping Spark session")


# ---------------------------------------
# File format extractors (lazy imports inside)
# ---------------------------------------
def extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (best) or pdfplumber fallback."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        fitz = None

    if fitz is not None:
        text = ""
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        pdf_document.close()
        return text

    # fallback
    try:
        import pdfplumber
    except Exception:
        pdfplumber = None

    if pdfplumber is not None:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    logger.error("No PDF extractor available. Install pymupdf (fitz) or pdfplumber.")
    return ""


def extract_from_docx(file_path: str) -> str:
    """Extract text from DOCX/DOC using python-docx."""
    try:
        import docx
    except Exception:
        logger.error("python-docx not installed. pip install python-docx")
        return ""

    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.exception(f"Error reading DOCX: {e}")
        return ""


def extract_from_txt(file_path: str) -> str:
    """Extract text from plain text files with encoding fallback."""
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    logger.error("Could not decode file using common encodings.")
    return ""


def extract_from_html(file_path: str) -> str:
    """Extract text from HTML using BeautifulSoup with fallback to regex tag removal."""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        BeautifulSoup = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n")
    else:
        logger.warning("beautifulsoup4 not installed, using simple regex to strip tags.")
        text = re.sub(r"<[^>]+>", "", html)
        return text


def extract_from_rtf(file_path: str) -> str:
    """Extract text from RTF using striprtf if available."""
    try:
        from striprtf.striprtf import rtf_to_text
    except Exception:
        rtf_to_text = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        rtf = f.read()

    if rtf_to_text is not None:
        try:
            return rtf_to_text(rtf)
        except Exception:
            logger.exception("striprtf failed to parse RTF.")
            return ""
    # fallback: naive cleanup
    text = re.sub(r"{\\.*?}", "", rtf, flags=re.DOTALL)
    text = re.sub(r"\\[a-zA-Z]+\s?", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def extract_from_odt(file_path: str) -> str:
    """Extract text from ODT using odfpy if available."""
    try:
        from odf import text as odftext, teletype
        from odf.opendocument import load
    except Exception:
        logger.error("odfpy not installed. pip install odfpy")
        return ""

    try:
        doc = load(file_path)
        paras = doc.getElementsByType(odftext.P)
        text = ""
        for p in paras:
            text += teletype.extractText(p) + "\n"
        return text
    except Exception:
        logger.exception("Error reading ODT")
        return ""


def extract_text(file_path: str) -> str:
    """
    Extract raw text from a supported file. Supported extensions:
     .pdf, .docx/.doc, .txt, .html/.htm, .md, .rtf, .odt
    """
    file_ext = Path(file_path).suffix.lower()
    logger.info(f"Extracting text from file type: {file_ext}")

    if file_ext == ".pdf":
        return extract_from_pdf(file_path)
    if file_ext in [".docx", ".doc"]:
        return extract_from_docx(file_path)
    if file_ext in [".txt", ".md"]:
        return extract_from_txt(file_path)
    if file_ext in [".html", ".htm"]:
        return extract_from_html(file_path)
    if file_ext == ".rtf":
        return extract_from_rtf(file_path)
    if file_ext == ".odt":
        return extract_from_odt(file_path)

    # fallback
    logger.warning("Unknown extension. Trying plain text read.")
    return extract_from_txt(file_path)


# ---------------------------------------
# Cleaning functions
# ---------------------------------------
def clean_text(text: str) -> str:
    """Basic text cleaning and normalization."""
    if not text:
        return ""

    # sanitize common control characters and non-breaking spaces
    text = text.replace("\x00", "").replace("\ufffd", "")
    text = text.replace("\xa0", " ")

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")

    # Remove page numbers and simple headers/footers by line
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*Chapter\s+\d+.*?$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # fix hyphenation across lines (word-\nnext -> wordnext)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # collapse excessive whitespace and multiple newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # normalize repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # remove empty lines and trim per-line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    return text


def remove_all_pii(text: str) -> str:
    """Redact simple PII patterns with placeholder tokens."""
    if not text:
        return ""

    # email
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)

    # phone numbers (common formats)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}", "[PHONE]", text)
    text = re.sub(r"\+\d{1,3}[-.\s]?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "[PHONE]", text)

    # credit-card-like numbers
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CREDIT_CARD]", text)

    # SSN-like
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)

    # IP addresses
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[IP]", text)

    # URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"www\.\S+", "[URL]", text)

    # dates common formats
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)

    # simple address pattern
    text = re.sub(
        r"\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b",
        "[ADDRESS]",
        text,
        flags=re.IGNORECASE,
    )

    # ZIP codes (US style)
    text = re.sub(r"\b\d{5}(?:-\d{4})?\b", "[ZIP]", text)

    # common name prefixes
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Professor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "[NAME]", text)

    return text


def remove_emojis(text: str) -> str:
    """Strip emoji and a set of ASCII emoticons."""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    emoticons = [":)", ":(", ":D", ":P", ";)", ":-)", ":-(", ":-D", ":-P", ";-)", "=)", "=(", "=D", ":o", ":-o", ":O", ":-O", ":|", ":-|"]
    for em in emoticons:
        text = text.replace(em, "")
    return text


def remove_html_tags(text: str) -> str:
    """Remove script/style tags and decode common HTML entities."""
    if not text:
        return ""
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove script/style
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", "", text)
    # common entities
    entities = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&copy;": "©",
        "&reg;": "®",
    }
    for k, v in entities.items():
        text = text.replace(k, v)
    # numeric entities
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    return text


def remove_cid_encoding(text: str) -> str:
    """Remove (cid:NNN) patterns and similar artifacts from PDFs."""
    if not text:
        return ""
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------
# Main pipeline function
# ---------------------------------------
def extract_and_clean(
    file_path: str,
    output_path: Optional[str] = None,
    remove_pii_flag: bool = True,
    spark=None,
    spark_app_name: str = "TextCleaning",
    spark_master: str = "local[*]",
) -> str:
    """
    Extract text from file_path and run cleaning pipeline. Returns cleaned text.

    Args:
      file_path: input path
      output_path: optional path to save cleaned text
      remove_pii_flag: whether to run PII redaction
      spark: optional SparkSession. If None, a local SparkSession will be created and stopped.
      spark_app_name, spark_master: used if we create a SparkSession.

    Returns:
      cleaned text (str)
    """
    # Lazy import Path to ensure module import is cheap
    file_ext = Path(file_path).suffix.lower()
    logger.info(f"Starting extraction for {file_path} (type {file_ext})")

    # Extract text
    raw_text = extract_text(file_path)
    logger.info(f"Extracted {len(raw_text)} chars")

    # Create Spark session if not provided
    created_spark = False
    if spark is None:
        try:
            spark = get_spark_session(app_name=spark_app_name, master=spark_master)
            created_spark = True
        except Exception as e:
            logger.warning("Could not create SparkSession; falling back to local processing. Error: %s", e)
            spark = None

    # If Spark is available, use RDD-based transforms for scale; otherwise fall back to local pipeline
    if spark is not None:
        rdd = spark.sparkContext.parallelize([raw_text], numSlices=1)

        if remove_pii_flag:
            rdd = rdd.map(remove_all_pii)

        rdd = rdd.map(remove_emojis)
        rdd = rdd.map(remove_html_tags)
        rdd = rdd.map(remove_cid_encoding)
        rdd = rdd.map(clean_text)

        cleaned = rdd.collect()[0]
    else:
        # local sequential processing
        cleaned = raw_text
        if remove_pii_flag:
            cleaned = remove_all_pii(cleaned)
        cleaned = remove_emojis(cleaned)
        cleaned = remove_html_tags(cleaned)
        cleaned = remove_cid_encoding(cleaned)
        cleaned = clean_text(cleaned)

    # Save if needed
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            logger.info("Saved cleaned output to %s", output_path)
        except Exception:
            logger.exception("Failed to write output to %s", output_path)

    # Stop Spark if we started it inside
    if created_spark and spark is not None:
        try:
            stop_spark_session(spark)
        except Exception:
            logger.exception("Error when stopping internal Spark session")

    logger.info("Finished cleaning: %d characters", len(cleaned))
    return cleaned


# # ---------------------------------------
# # CLI / demo
# if __name__ == "__main__":
#     # Change this path for quick local testing
#     input_file = "/content/hepr101.pdf"
#     output_file = "cleaned_output.txt"

#     cleaned_text = extract_and_clean(
#         file_path=input_file,
#         output_path=output_file,
#         remove_pii_flag=True
#     )

#     print("Cleaning completed.")
