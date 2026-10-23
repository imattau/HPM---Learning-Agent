"""Convert local books, wiki exports, conversations, web pages, and code to HFN source JSONL.

This utility prepares corpus material; it does not fetch data or grant reuse rights.
Keep source URLs, license terms, and collection notes in the emitted metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".text", ".py", ".ipynb", ".js", ".jsx",
    ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".go", ".rs",
    ".sh", ".sql", ".yaml", ".yml", ".toml", ".ini", ".css", ".html",
    ".htm", ".xhtml", ".xml", ".json", ".jsonl", ".epub", ".pdf",
}
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".go", ".rs", ".sh", ".sql", ".yaml", ".yml", ".toml", ".ini",
}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml", ".xml"}
BOOK_EXTENSIONS = {".epub", ".pdf"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return _normalise(" ".join(parser.parts))


def _normalise(value: str) -> str:
    value = html.unescape(value).replace("\x00", " ")
    return re.sub(r"[\t\r\f\v ]+", " ", re.sub(r"\n\s*\n+", "\n\n", value)).strip()


def _text_from_record(record: Any) -> tuple[str, str]:
    if isinstance(record, str):
        return record, ""
    if not isinstance(record, dict):
        return "", ""
    title = str(record.get("title") or record.get("name") or record.get("id") or "")
    for key in ("text", "content", "body", "article", "transcript"):
        if isinstance(record.get(key), str) and record[key].strip():
            return record[key], title
    messages = record.get("messages") or record.get("conversation") or record.get("turns")
    if isinstance(messages, list):
        turns = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("speaker") or "").strip()
            content = message.get("content") or message.get("text") or message.get("message")
            if isinstance(content, str) and content.strip():
                turns.append(f"{role}: {content.strip()}" if role else content.strip())
        return "\n".join(turns), title
    return "", title


def _structured_text(value: Any) -> str:
    """Extract prose from nested Wikipedia-style section payloads."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _structured_text(item)))
    if isinstance(value, dict):
        heading = next((
            value[key].strip() for key in ("heading", "title")
            if isinstance(value.get(key), str) and value[key].strip()
        ), "")
        body = next((
            value[key].strip() for key in ("text", "content", "plain_text")
            if isinstance(value.get(key), str) and value[key].strip()
        ), "")
        if heading or body:
            return f"{heading}: {body}" if heading and body else heading or body
        return "\n".join(
            part for item in value.values() if (part := _structured_text(item))
        )
    return ""


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"], check=True,
                capture_output=True, text=True,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"PDF ingestion requires pypdf or the pdftotext command: {path}"
            ) from exc
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _extract(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [(_pdf_text(path), path.stem)]
    if suffix == ".epub":
        with zipfile.ZipFile(path) as archive:
            documents = []
            for name in sorted(n for n in archive.namelist() if n.lower().endswith((".html", ".xhtml"))):
                text = html_to_text(archive.read(name).decode("utf-8", errors="replace"))
                if text:
                    documents.append((text, f"{path.stem} — {Path(name).stem}"))
        return documents
    if suffix in {".json", ".jsonl", ".ipynb"}:
        raw = json.loads(path.read_text(encoding="utf-8")) if suffix != ".jsonl" else None
        if suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".ipynb":
            notebook = raw if isinstance(raw, dict) else {}
            cells = notebook.get("cells", [])
            records = [{"title": path.stem, "text": "\n".join(
                f"[{cell.get('cell_type', 'cell')}] {''.join(cell.get('source', []))}"
                for cell in cells if isinstance(cell, dict)
            )}]
        else:
            records = raw if isinstance(raw, list) else [raw]
        return [(_normalise(text), title or path.stem) for text, title in map(_text_from_record, records) if text.strip()]
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_EXTENSIONS:
        text = html_to_text(text)
    return [(text, path.stem)] if text.strip() else []


def _source_type(path: Path, override: str | None) -> str:
    if override:
        return override
    labels = {part.casefold() for part in path.parts}
    suffix = path.suffix.lower()
    if labels & {"conversation", "conversations", "chat", "chats", "dialogue"}:
        return "conversation"
    if labels & {"wiki", "wikipedia", "wikimedia"}:
        return "wiki"
    if labels & {"book", "books", "ebooks", "library"} or suffix in BOOK_EXTENSIONS:
        return "book"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in HTML_EXTENSIONS or labels & {"web", "websites", "webpages"}:
        return "website"
    return "text"


def _load_huggingface_records(
    dataset_id: str, *, config: str | None, split: str, revision: str | None,
    license_name: str, max_rows: int, streaming: bool,
    text_columns: tuple[str, ...], source_type: str | None,
    language_filter: str | None, license_column: str,
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face ingestion requires the `datasets` package") from exc
    try:
        dataset = load_dataset(
            dataset_id, name=config, split=split, revision=revision, streaming=streaming,
        )
    except Exception as original_error:
        # Some dataset repositories publish static Parquet splits whose Hub
        # metadata disagrees with the cached dataset card. Load those data files
        # directly, without executing dataset-provided code.
        try:
            from huggingface_hub import HfApi, hf_hub_download
            files = HfApi().list_repo_files(
                dataset_id, repo_type="dataset", revision=revision,
            )
            split_files = [
                filename for filename in files
                if filename.lower().endswith(".parquet")
                and split in Path(filename).name.casefold()
            ]
            if not split_files:
                raise original_error
            local_files = [hf_hub_download(
                dataset_id, filename=filename, repo_type="dataset", revision=revision,
            ) for filename in split_files]
            dataset = load_dataset(
                "parquet", data_files={split: local_files}, split=split,
                streaming=streaming,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"could not load Hugging Face dataset {dataset_id}@{revision}, split={split}"
            ) from fallback_error
    accepted = 0
    for index, row in enumerate(dataset):
        if not isinstance(row, dict):
            continue
        if language_filter and str(row.get("lang") or row.get("language") or "").casefold() != language_filter.casefold():
            continue
        text_parts = []
        title = str(row.get("title") or row.get("name") or row.get("id") or f"row-{index}")
        for column in text_columns:
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                if column in {"sections", "article_sections"}:
                    extracted = _structured_text(value)
                    if extracted:
                        text_parts.append(extracted)
                else:
                    text_parts.append(value.strip())
            elif column in {"sections", "article_sections"} and isinstance(value, (list, dict)):
                extracted = _structured_text(value)
                if extracted:
                    text_parts.append(extracted)
            elif column in {"messages", "conversation", "turns"} and isinstance(value, list):
                turns, _ = _text_from_record({"messages": value})
                if turns:
                    text_parts.append(turns)
        if not text_parts:
            value, fallback_title = _text_from_record(row)
            if value:
                text_parts.append(value)
                title = fallback_title or title
        text = "\n".join(text_parts).strip()
        if not text:
            continue
        row_license = row.get(license_column)
        if license_name == "per-row" and not (isinstance(row_license, str) and row_license.strip()):
            raise ValueError(
                f"Hugging Face dataset {dataset_id} row {index} has no row-level license; "
                "cannot import it with --license per-row"
            )
        role = row.get("role") or row.get("speaker")
        if role and source_type == "conversation":
            text = f"{role}: {text}"
        metadata = {
            "huggingface_dataset_id": dataset_id,
            "huggingface_config": config,
            "huggingface_revision": revision or "default",
            "huggingface_split": split,
            "huggingface_row_id": str(row.get("id") or row.get("message_id") or index),
            "source_url": row.get("url") or row.get("source_url"),
            "license": row_license or license_name,
            "license_source_column": license_column if row_license else "dataset-card",
            "parent_id": row.get("parent_id"),
            "conversation_id": row.get("message_tree_id") or row.get("conversation_id"),
            "role": row.get("role") or row.get("speaker"),
            "language": row.get("lang") or row.get("language"),
        }
        yield text, title, metadata
        accepted += 1
        if accepted >= max_rows:
            break


def _chunks(text: str, max_chars: int) -> Iterable[str]:
    text = _normalise(text)
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = text.rfind(". ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        else:
            cut += 1
        chunk, text = text[:cut].strip(), text[cut:].strip()
        if chunk:
            yield chunk
    if text:
        yield text


def prepare_corpus(
    inputs: Iterable[str | Path] = (), *, output: str | Path,
    license_name: str = "unspecified", source_type: str | None = None,
    max_chars: int = 5000, hf_dataset: str | None = None,
    hf_config: str | None = None, hf_split: str = "train",
    hf_revision: str | None = None, hf_max_rows: int = 10_000,
    hf_streaming: bool = False, hf_text_columns: Iterable[str] = ("text", "content", "messages"),
    hf_language: str | None = None, hf_license_column: str = "license",
) -> dict[str, Any]:
    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    if hf_max_rows < 1:
        raise ValueError("hf_max_rows must be positive")
    if hf_dataset and license_name == "unspecified":
        raise ValueError("state the Hugging Face dataset's declared license with --license")
    if hf_dataset and not hf_revision:
        raise ValueError("pin a Hugging Face dataset commit or tag with --hf-revision")
    if not inputs and not hf_dataset:
        raise ValueError("provide local input paths, --hf-dataset, or both")
    roots = [Path(item) for item in inputs]
    files: list[Path] = []
    excluded_parts = {".git", ".venv", "__pycache__", "results", "node_modules"}
    for root in roots:
        files.extend([root] if root.is_file() else sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
            and not (set(p.parts) & excluded_parts)
        ))
    rows = []
    counts: dict[str, int] = {}
    for path in files:
        kind = _source_type(path, source_type)
        for doc_index, (text, title) in enumerate(_extract(path)):
            for chunk_index, chunk in enumerate(_chunks(text, max_chars)):
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                source_id = f"{kind}-{digest[:20]}"
                metadata = {
                    "source_type": kind,
                    "source_path": str(path),
                    "license": license_name,
                    "content_sha256": digest,
                    "document_index": doc_index,
                    "chunk_index": chunk_index,
                }
                rows.append({
                    "kind": "source", "source_id": source_id, "text": chunk,
                    "title": title, "metadata": metadata,
                })
    hf_row_count = 0
    if hf_dataset:
        hf_kind = source_type or "text"
        for row_id, (text, title, hf_metadata) in enumerate(_load_huggingface_records(
            hf_dataset, config=hf_config, split=hf_split, revision=hf_revision,
            license_name=license_name, max_rows=hf_max_rows, streaming=hf_streaming,
            text_columns=tuple(hf_text_columns), source_type=source_type,
            language_filter=hf_language, license_column=hf_license_column,
        )):
            hf_row_count += 1
            for chunk_index, chunk in enumerate(_chunks(text, max_chars)):
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                identity = "|".join((
                    hf_dataset, hf_revision or "", hf_split,
                    hf_metadata.get("huggingface_row_id", str(row_id)), str(chunk_index),
                ))
                source_id = f"{hf_kind}-hf-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
                rows.append({
                    "kind": "source", "source_id": source_id,
                    "text": chunk, "title": title,
                    "metadata": {
                        **hf_metadata, "source_type": hf_kind,
                        "content_sha256": digest, "chunk_index": chunk_index,
                    },
                })
    # Deduplicate text, but retain every distinct origin for provenance audits.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = unique.get(row["source_id"])
        if previous is None:
            unique[row["source_id"]] = row
        else:
            origins = previous["metadata"].setdefault("duplicate_origins", [])
            origins.append({
                key: value for key, value in row["metadata"].items()
                if key not in {"content_sha256", "source_type"}
            })
    for row in unique.values():
        kind = row["metadata"].get("source_type", "unspecified")
        counts[kind] = counts.get(kind, 0) + 1
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique.values()), encoding="utf-8")
    return {
        "output": str(out_path), "input_file_count": len(files),
        "huggingface_dataset": hf_dataset, "huggingface_rows_read": hf_row_count,
        "source_document_count": len(unique),
        "characters": sum(len(row["text"]) for row in unique.values()),
        "source_type_counts": counts,
        "license": license_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--license", default="unspecified", dest="license_name")
    parser.add_argument("--source-type", choices=("book", "wiki", "conversation", "website", "code", "text"))
    parser.add_argument("--max-chars", type=int, default=5000)
    parser.add_argument("--hf-dataset", help="Hugging Face dataset repository ID")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-revision", help="Pin a dataset commit or tag")
    parser.add_argument("--hf-max-rows", type=int, default=10_000)
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--hf-language", help="keep rows matching this language code")
    parser.add_argument("--hf-license-column", default="license", help="row field holding its license, for per-row license datasets")
    parser.add_argument("--hf-text-column", action="append", dest="hf_text_columns")
    args = parser.parse_args()
    print(json.dumps(prepare_corpus(
        args.inputs, output=args.output, license_name=args.license_name,
        source_type=args.source_type, max_chars=args.max_chars,
        hf_dataset=args.hf_dataset, hf_config=args.hf_config,
        hf_split=args.hf_split, hf_revision=args.hf_revision,
        hf_max_rows=args.hf_max_rows, hf_streaming=args.hf_streaming,
        hf_text_columns=args.hf_text_columns or ("text", "content", "messages"),
        hf_language=args.hf_language, hf_license_column=args.hf_license_column,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
