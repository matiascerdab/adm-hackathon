
import hashlib
import hmac
import json
import re
import secrets
from pathlib import Path

DATA_ROOT = Path(r"C:\adm-hackathon-data")

PROCESSED_DIR = DATA_ROOT / "processed"
CLEAN_DIR = DATA_ROOT / "clean"
PRIVATE_DIR = DATA_ROOT / "private"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = CLEAN_DIR / "corpus.jsonl"
MAPPING_PATH = PRIVATE_DIR / "document_mapping.jsonl"
REPORT_PATH = PRIVATE_DIR / "cleaning_report.json"
MANIFEST_PATH = PRIVATE_DIR / "source_manifest.sha256"
KEY_PATH = PRIVATE_DIR / "document_id_key.bin"


def load_or_create_key():
    """Keep document IDs stable across repeated runs."""

    if KEY_PATH.exists():
        key = KEY_PATH.read_bytes()
        if len(key) != 32:
            raise ValueError("Invalid document ID key")
        return key

    key = secrets.token_bytes(32)
    KEY_PATH.write_bytes(key)
    return key


def generate_doc_id(original_id, key):
    """Generate a nonsequential pseudorandom document ID."""

    digest = hmac.new(
        key,
        str(original_id).encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return "doc_" + digest[:20]


def clean_text(text):
    """Apply conservative text normalization."""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_dataset():

    key = load_or_create_key()

    files = sorted(PROCESSED_DIR.glob("*.json"))

    if not files:
        raise RuntimeError("No processed JSON documents found")

    seen_text_hashes = set()
    seen_doc_ids = set()

    stats = {
        "input_files": len(files),
        "accepted": 0,
        "exact_duplicates": 0,
        "empty_text": 0,
        "invalid_records": 0
    }

    with (
        CORPUS_PATH.open("w", encoding="utf-8") as corpus_file,
        MAPPING_PATH.open("w", encoding="utf-8") as mapping_file,
        MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file
    ):

        for path in files:

            raw_bytes = path.read_bytes()

            source_hash = hashlib.sha256(
                raw_bytes
            ).hexdigest()

            manifest_file.write(
                f"{source_hash}  {path.name}\n"
            )

            try:
                document = json.loads(raw_bytes)

                original_id = str(document["document_id"])
                text = clean_text(document["text"])

                if original_id != path.stem:
                    raise ValueError("Filename and ID mismatch")

                if not text:
                    stats["empty_text"] += 1
                    continue

                text_hash = hashlib.sha256(
                    text.casefold().encode("utf-8")
                ).hexdigest()

                if text_hash in seen_text_hashes:
                    stats["exact_duplicates"] += 1
                    continue

                doc_id = generate_doc_id(original_id, key)

                if doc_id in seen_doc_ids:
                    raise ValueError("Document ID collision")

                seen_text_hashes.add(text_hash)
                seen_doc_ids.add(doc_id)

                clean_record = {
                    "doc_id": doc_id,
                    "text": text
                }

                private_record = {
                    "doc_id": doc_id,
                    "wikipedia_id": original_id,
                    "title": document.get("title"),
                    "source_category": document.get("source_category"),
                    "source_topic": document.get("source_topic"),
                    "source_url": document.get("source_url"),
                    "revision_id": document.get("revision_id"),
                    "revision_timestamp": document.get(
                        "revision_timestamp"
                    ),
                    "retrieved_at": document.get("retrieved_at")
                }

                corpus_file.write(
                    json.dumps(clean_record, ensure_ascii=False) + "\n"
                )

                mapping_file.write(
                    json.dumps(private_record, ensure_ascii=False) + "\n"
                )

                stats["accepted"] += 1

            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                stats["invalid_records"] += 1

    REPORT_PATH.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    print("\nDataset construction completed")
    print("--------------------------------")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Private mapping: {MAPPING_PATH}")


if __name__ == "__main__":
    build_dataset()
