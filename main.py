#!/usr/bin/env python3
import os, re, json, time, pathlib, argparse, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import Counter
import requests
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# PDF deps
import pdfplumber
from pypdf import PdfReader

# EPUB deps
from ebooklib import epub
from bs4 import BeautifulSoup, Tag

# ---------------- Config ----------------
DEFAULT_MODEL = "gpt-oss:20b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://100.85.236.35:11434")
MAX_CHUNK_CHARS = 12000
CHUNK_OVERLAP = 800
OUT_DIR = "summaries"
HEADING_NEARBY_SCAN_PAGES = 3
EPUB_PAGE_SIZE_DEFAULT = 3000  # characters per synthetic EPUB page

console = Console()

# --- Summarization style: THOROUGH & LINEAR (no quotes, bullets, questions)
SYSTEM_SUMMARY = """You are a careful book summarizer.
Produce a thorough, linear summary that preserves the story flow and all important details (characters, events, arguments, evidence).
Do NOT include quotes, bullet points, Q&A, or discussion questions. No meta commentary.
Write in clear paragraphs, in the order the chapter proceeds, capturing all key developments and transitions."""

# --- TOC inference prompts
SYSTEM_TOC_PDF = """You will see the first pages of a book PDF (often containing the table of contents).
Infer the chapter list and starting PDF page numbers (1-indexed in the PDF viewer). Include chapter titles exactly as seen when possible.
Return ONLY JSON (no commentary) with this schema:
[
  {"title": "Chapter 1: <Title>", "start_page": 3},
  {"title": "Chapter 2: <Title>", "start_page": 15}
]
If no explicit index is visible, infer likely chapter titles and starts if possible; otherwise return an empty list.
"""

SYSTEM_TOC_EPUB = """You will see the opening sections of an EPUB (HTML content). Infer the chapter list and the starting section indices.
Treat each EPUB spine section as one unit. Indices are 1-indexed by spine order.
Return ONLY JSON (no commentary) with this schema:
[
  {"title": "Chapter 1: <Title>", "start_section": 1},
  {"title": "Chapter 2: <Title>", "start_section": 7}
]
If no explicit index is visible, infer likely chapters if possible; otherwise return an empty list.
"""

@dataclass
class Chapter:
    idx_start: int      # 0-based start index (PDF page idx, EPUB spine idx, or synthetic page idx)
    idx_end: Optional[int]  # exclusive for PDF/EPUB-spine; for synthetic pages we treat as inclusive page number
    title: str
    text: str = ""

# ---------------- Ollama ----------------
def ollama_generate(prompt: str, model: str, stream: bool = True, options: Optional[dict] = None) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if options:
        payload["options"] = options
    r = requests.post(url, json=payload, stream=stream, timeout=600)
    r.raise_for_status()
    if not stream:
        return r.json().get("response", "")
    out = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if "response" in obj:
            out.append(obj["response"])
    return "".join(out)

# ---------------- Utilities ----------------
def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + 0.4 * max_chars:
            boundary = end
        chunks.append(text[start:boundary].strip())
        if boundary == len(text):
            break
        start = max(0, boundary - overlap_chars)
    deduped = []
    for c in chunks:
        if not deduped or c not in deduped[-1]:
            deduped.append(c)
    return deduped

def summarize_chunk(chapter_title: str, chunk_text_: str, model: str) -> str:
    prompt = f"""System:
{SYSTEM_SUMMARY}

User:
You are summarizing a single PART of a chapter from a book. Chapter title: "{chapter_title}".

Text to summarize (verbatim):
\"\"\"{chunk_text_}\"\"\"

Write a thorough, linear paragraph-style summary of ONLY this chunk. No quotes, no bullets, no questions.
Write only what happens exactly in the story and leave out the unnecessary detail and concentrate on the storyline.
"""
    return ollama_generate(prompt, model=model)

def merge_chunk_summaries(chapter_title: str, parts: List[str], model: str) -> str:
    joined = "\n\n".join(parts)
    prompt = f"""System:
{SYSTEM_SUMMARY}

User:
Below are partial summaries for different chunks of the SAME chapter titled "{chapter_title}".
Merge them into a single coherent chapter summary in paragraphs only (no bullets, no quotes, no questions).

Partial summaries:
\"\"\"{joined}\"\"\"
"""
    return ollama_generate(prompt, model=model)

# ---------------- PDF loader ----------------
PDF_HEADING_REGEXES = [
    r"^\s*(chapter|chap\.?)\s+\d+[\.:]?\s+.*$",
    r"^\s*chapter\s+\d+\s*$",
    r"^\s*[ivxlcdm]+\.\s+.+$",
    r"^\s*\d+\.\s+.+$",
    r"^\s*[A-Z][A-Z0-9\-\s]{5,}$",
]
def _compile_pdf_heading_regexes():
    return [re.compile(rx, re.IGNORECASE) for rx in PDF_HEADING_REGEXES]

def pdf_extract_pages(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages

def pdf_try_outline(pdf_path: str) -> List[Tuple[int, Optional[int], str]]:
    reader = PdfReader(pdf_path)
    try:
        toc = reader.outline
    except Exception:
        toc = []
    def iter_items(items, level=0):
        for it in items:
            if isinstance(it, list):
                yield from iter_items(it, level+1)
            else:
                title = getattr(it, "title", None) or str(it)
                try:
                    page = reader.get_destination_page_number(it)
                except Exception:
                    page = None
                yield {"title": title, "page": page, "level": level}
    flat = [x for x in iter_items(toc, 0) if x.get("page") is not None]
    top = [x for x in flat if x["level"] == 0]
    top_sorted = sorted(top, key=lambda d: d["page"])
    out = []
    for i, item in enumerate(top_sorted):
        start = item["page"]
        end = top_sorted[i+1]["page"] if i+1 < len(top_sorted) else None
        title = re.sub(r"\s+", " ", item["title"]).strip()
        out.append((start, end, title))
    return out

def pdf_find_heading_near(pages: List[str], guess_idx: int) -> Optional[int]:
    compiled = _compile_pdf_heading_regexes()
    low = max(0, guess_idx - HEADING_NEARBY_SCAN_PAGES)
    high = min(len(pages) - 1, guess_idx + HEADING_NEARBY_SCAN_PAGES)
    for i in range(low, high + 1):
        lines = [l for l in (pages[i].splitlines() if pages[i] else []) if l.strip()]
        head = lines[:15]
        for line in head:
            if any(rx.search(line) for rx in compiled):
                return i
    return None

def pdf_slice_pages(pages: List[str], start: int, end: Optional[int]) -> str:
    if end is None:
        end = len(pages)
    return "\n\n".join(pages[start:end])

# ---------------- EPUB loader ----------------
def epub_load(path: str):
    return epub.read_epub(path)

def epub_spine_texts(book) -> List[str]:
    """Extract visible text from each spine item in reading order."""
    spine_texts: List[str] = []
    for (idref, _) in book.spine:
        item = book.get_item_with_id(idref)
        # Some versions of ebooklib don't expose ITEM_DOCUMENT; check MIME type instead
        if item is None or not item.media_type.startswith("application/xhtml"):
            spine_texts.append("")
            continue

        soup = BeautifulSoup(item.get_body_content(), "lxml")

        # Keep readable spacing
        for br in soup.find_all(["br"]):
            br.replace_with("\n")
        text = soup.get_text("\n")

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        spine_texts.append(text.strip())
    return spine_texts

def epub_try_outline(book) -> List[Tuple[int, Optional[int], str]]:
    """
    Build a top-level outline from the EPUB TOC (nav) that follows links:
    - resolves fragments (#anchors)
    - resolves relative paths
    - maps TOC entries to spine items even if the href isn't a direct spine file
    """

    # --- Build maps ---
    spine_map = []           # [(clean_path, index)]
    path_to_spine = {}        # clean_path → index

    for i, (idref, _) in enumerate(book.spine):
        item = book.get_item_with_id(idref)
        if not item:
            continue
        name = item.get_name()  # e.g. "Text/ch1.xhtml"
        clean = name.strip().lstrip("/")
        spine_map.append((clean, i))
        path_to_spine[clean] = i

    def normalize_href(href: str) -> Tuple[str, str]:
        """Return (path_without_fragment, fragment)"""
        if not href:
            return "", ""
        parts = href.split("#", 1)
        base = parts[0].strip().lstrip("/")
        frag = parts[1] if len(parts) > 1 else ""
        return base, frag

    # --- Flatten TOC entries ---
    def flatten(toc, level=0):
        for entry in toc:
            if isinstance(entry, epub.Link):
                yield {"title": entry.title, "href": entry.href, "level": level}
            elif isinstance(entry, tuple) and len(entry) >= 2:
                item, children = entry[0], entry[1]
                if isinstance(item, epub.Link):
                    yield {"title": item.title, "href": item.href, "level": level}
                for sub in flatten(children, level+1):
                    yield sub

    flat = list(flatten(book.toc, 0))

    matched = []

    for ent in flat:
        href = ent.get("href", "")
        title = ent.get("title", "").strip()
        base, frag = normalize_href(href)

        # --- Direct spine match ---
        if base in path_to_spine:
            matched.append({"title": title, "spine": path_to_spine[base], "level": ent["level"]})
            continue

        # --- Try relative/loose matching (very common) ---
        # e.g. nav.xhtml linking to "ch1.xhtml" while spine uses "Text/ch1.xhtml"
        for clean, idx in spine_map:
            if clean.endswith(base):
                matched.append({"title": title, "spine": idx, "level": ent["level"]})
                break
        else:
            # --- LAST RESORT: scan spine text to find anchor ---
            if frag:
                anchor_pattern = re.compile(rf'id\s*=\s*"{re.escape(frag)}"', re.IGNORECASE)
                for clean, idx in spine_map:
                    item = book.get_item_with_id(book.spine[idx][0])
                    if not item:
                        continue
                    html = item.get_content().decode(errors="ignore")
                    if anchor_pattern.search(html):
                        matched.append({"title": title, "spine": idx, "level": ent["level"]})
                        break

    # Take top-level entries only
    top = [x for x in matched if x["level"] == 0]
    top_sorted = sorted(top, key=lambda d: d["spine"])

    # Build contiguous chapter ranges
    out = []
    for i, it in enumerate(top_sorted):
        start = it["spine"]
        end = top_sorted[i+1]["spine"] if i+1 < len(top_sorted) else None
        title = re.sub(r"\s+", " ", it["title"]).strip()
        out.append((start, end, title))

    return out

def epub_slice_sections(sections: List[str], start: int, end: Optional[int]) -> str:
    if end is None:
        end = len(sections)
    return "\n\n".join(sections[start:end])

# ---------------- EPUB heading-based synthetic pages ----------------
def clean_epub_text(s: str) -> str:
    """Collapse whitespace for plain-text strings."""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_section_text_from_heading(heading: Tag, heading_names) -> str:
    """
    Given a heading tag (h1/h2/h3), collect all text from it
    until the next heading tag of any of those names.
    """
    parts = []

    # Include heading text as part of the section
    heading_text = heading.get_text(separator=" ", strip=True)
    if heading_text:
        parts.append(heading_text)

    # Then all following siblings until next heading
    for elem in heading.next_siblings:
        if isinstance(elem, Tag) and elem.name in heading_names:
            break  # next chapter begins
        if isinstance(elem, Tag):
            t = elem.get_text(separator=" ", strip=True)
        else:
            t = str(elem).strip()
        if t:
            parts.append(t)

    return clean_epub_text(" ".join(parts))

def extract_chapters_from_epub_item(item, chapter_index_offset: int, used_titles: Counter):
    """
    From a single XHTML item, extract a list of (chapter_title, text)
    using H1/H2/H3 as chapter boundaries.

    chapter_index_offset is used for fallback "Chapter N" numbering.

    Returns:
        chapters: list[(title, text)]
        new_offset: updated offset for next fallback index
    """
    html = item.get_content()
    soup = BeautifulSoup(html, "html.parser")

    heading_names = ["h1", "h2", "h3"]
    headings = [h for h in soup.find_all(heading_names) if h.get_text(strip=True)]

    chapters = []
    local_index = 0

    for h in headings:
        # Proposed title: heading text
        title = h.get_text(separator=" ", strip=True)
        if not title:
            title = f"Chapter {chapter_index_offset + local_index + 1}"

        # Make titles unique
        used_titles[title] += 1
        count = used_titles[title]
        if count > 1:
            unique_title = f"{title} ({count})"
        else:
            unique_title = title

        text = extract_section_text_from_heading(h, heading_names)
        if text:
            chapters.append((unique_title, text))
            local_index += 1

    # If no headings at all, treat whole file as one chapter
    if not chapters:
        full_text = soup.get_text(separator=" ", strip=True)
        full_text = clean_epub_text(full_text)
        if full_text:
            fallback_title = f"Chapter {chapter_index_offset + 1}"
            used_titles[fallback_title] += 1
            count = used_titles[fallback_title]
            if count > 1:
                fallback_title = f"{fallback_title} ({count})"
            chapters.append((fallback_title, full_text))
            local_index += 1

    return chapters, chapter_index_offset + local_index

def epub_build_chapters_from_headings(epub_path: str, page_size: int) -> List[Chapter]:
    """
    Build chapters from EPUB by:
    - Treating each h1/h2/h3 as a chapter boundary.
    - Computing synthetic page ranges based on character offsets.

    Returns a list of Chapter where:
      - idx_start: 0-based synthetic start page index
      - idx_end:   synthetic end page (inclusive)
      - title, text: full chapter text
    """
    book = epub.read_epub(epub_path)
    spine_ids = [item_id for (item_id, _) in book.spine]

    used_titles = Counter()

    # 1) Extract all chapters in strict reading order
    all_chapters: List[Tuple[str, str]] = []  # (title, text)
    chapter_index_offset = 0

    for item_id in spine_ids:
        item = book.get_item_with_id(item_id)
        # Only process HTML/XHTML-like items
        if item is None or not getattr(item, "media_type", "").startswith("application/xhtml"):
            continue

        chapters, chapter_index_offset = extract_chapters_from_epub_item(
            item, chapter_index_offset, used_titles
        )
        all_chapters.extend(chapters)

    if not all_chapters:
        return []

    # 2) Convert chapters to synthetic page ranges based on global char offset
    results: List[Chapter] = []
    offset = 0

    for title, text in all_chapters:
        length = len(text)
        if length == 0:
            continue

        start_offset = offset
        end_offset = offset + length - 1

        start_page = start_offset // page_size + 1
        end_page = end_offset // page_size + 1

        # Store pages in idx_* so the rest of the pipeline can reuse them.
        # idx_start is 0-based page index; idx_end is inclusive page number.
        ch = Chapter(
            idx_start=start_page - 1,
            idx_end=end_page,
            title=title,
            text=text,
        )
        results.append(ch)
        offset += length

    return results

# ---------------- LLM TOC inference ----------------
def ask_llm_for_toc(pieces: List[str], model: str, mode: str) -> List[Dict[str, Any]]:
    """
    mode: 'pdf' uses SYSTEM_TOC_PDF and expects start_page
          'epub' uses SYSTEM_TOC_EPUB and expects start_section
    """
    snippet = "\n\n".join(pieces)
    system = SYSTEM_TOC_PDF if mode == "pdf" else SYSTEM_TOC_EPUB
    prompt = f"""System:
{system}

User:
Here is the opening content:

\"\"\"{snippet[:120000]}\"\"\"

Return ONLY JSON as specified."""
    console.print(Panel.fit("Querying LLM to infer chapters…", style="bold cyan"))
    raw = ollama_generate(prompt, model=model)
    json_text = raw.strip()
    first = json_text.find("[")
    last = json_text.rfind("]")
    if first != -1 and last != -1:
        json_text = json_text[first:last+1]
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    console.print("[yellow]LLM did not return clean JSON; treating as no TOC found.[/yellow]")
    return []

# ---------------- Manual TOC (author input) ----------------
MANUAL_TOC_HELP = """
No index detected. Please provide a chapter list so we can proceed.

For PDF, use:  {"title": "...", "start_page": 5}
For EPUB, use: {"title": "...", "start_section": 3}

Paste JSON exactly like:
[
  {"title": "Chapter 1: Introduction", "start_page": 5},
  {"title": "Chapter 2: Methods", "start_page": 23}
]

or

[
  {"title": "Chapter 1: Introduction", "start_section": 1},
  {"title": "Chapter 2: Methods", "start_section": 7}
]

Notes:
- PDF start_page is 1-indexed (as shown in your PDF viewer).
- EPUB start_section is 1-indexed by spine order (first content file = 1).
- Include every chapter in reading order.
- Leave a blank line and press Ctrl-D (macOS/Linux) or Ctrl-Z then Enter (Windows) when done.
"""

def request_manual_toc_from_stdin() -> List[Dict[str, Any]]:
    console.print(Panel.fit(MANUAL_TOC_HELP.strip(), style="bold yellow"))
    console.print("Paste JSON now, then submit (EOF):", style="yellow")
    buf = sys.stdin.read()
    try:
        data = json.loads(buf)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    console.print("[red]Invalid JSON provided. Aborting.[/red]")
    sys.exit(2)

def load_manual_toc_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manual TOC file must contain a JSON list.")
    return data

# ---------------- Build chapters (PDF) ----------------
def build_chapters_pdf(path: str, lookahead_pages: int, model: str,
                       manual_toc: Optional[List[Dict[str, Any]]]) -> List[Chapter]:
    pages = pdf_extract_pages(path)

    console.rule("[bold]Stage 1 (PDF): Try native PDF outline[/bold]")
    outline = pdf_try_outline(path)
    if outline:
        console.print(f"[green]Found outline with {len(outline)} entries.[/green]")
        chapters = []
        for (start, end, title) in outline:
            text = pdf_slice_pages(pages, start, end)
            chapters.append(Chapter(idx_start=start, idx_end=end, title=title, text=text))
        return chapters

    console.print("[yellow]No usable outline. Using LLM to infer chapters.[/yellow]")
    console.rule("[bold]Stage 2 (PDF): LLM TOC inference from first pages[/bold]")
    toc_guess = ask_llm_for_toc(pages[:lookahead_pages], model=model, mode="pdf")

    if not toc_guess:
        toc_guess = manual_toc if manual_toc else request_manual_toc_from_stdin()

    console.rule("[bold]Stage 3 (PDF): Align to headings[/bold]")
    starts: List[int] = []
    titles_by_idx: Dict[int, str] = {}
    for item in toc_guess:
        sp = item.get("start_page")
        if isinstance(sp, int) and sp >= 1:
            guess_idx = min(len(pages)-1, max(0, sp - 1))
            aligned = pdf_find_heading_near(pages, guess_idx)
            idx = aligned if aligned is not None else guess_idx
            starts.append(idx)
            titles_by_idx[idx] = (str(item.get("title") or "").strip() or f"Chapter starting at page {idx+1}")
    if 0 not in starts:
        starts.append(0)
        titles_by_idx.setdefault(0, "Chapter starting at page 1")
    starts = sorted(set(starts))

    chapters: List[Chapter] = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else None
        title = titles_by_idx.get(s)
        if not title:
            # fallback title from page text
            lines = [l.strip() for l in (pages[s].splitlines() if pages[s] else []) if l.strip()]
            title = (lines[0][:120] if lines else f"Chapter starting at page {s+1}")
        text = pdf_slice_pages(pages, s, e)
        chapters.append(Chapter(idx_start=s, idx_end=e, title=title, text=text))

    # Show alignment
    table = Table(title="Aligned Chapters (PDF)", show_lines=True)
    table.add_column("#", justify="right"); table.add_column("Title")
    table.add_column("Start PDF page (idx+1)"); table.add_column("End page idx")
    for i, ch in enumerate(chapters, start=1):
        table.add_row(str(i), ch.title, str(ch.idx_start+1), str(ch.idx_end or len(pages)))
    console.print(table)
    return chapters

# ---------------- Build chapters (EPUB) ----------------
def build_chapters_epub(path: str, lookahead_sections: int, model: str,
                        manual_toc: Optional[List[Dict[str, Any]]],
                        epub_page_size: int) -> List[Chapter]:
    # Stage 0: try pure heading-based chapter detection + synthetic pages
    console.rule("[bold]Stage 0 (EPUB): Heading-based synthetic pages[/bold]")
    heading_chapters = epub_build_chapters_from_headings(path, epub_page_size)
    if heading_chapters:
        console.print(
            f"[green]Built {len(heading_chapters)} chapters from HTML headings "
            f"using page size {epub_page_size} chars.[/green]"
        )
        table = Table(title="Chapters from headings (EPUB)", show_lines=True)
        table.add_column("#", justify="right")
        table.add_column("Title")
        table.add_column("Start synthetic page")
        table.add_column("End synthetic page")

        for i, ch in enumerate(heading_chapters, start=1):
            table.add_row(str(i), ch.title, str(ch.idx_start + 1), str(ch.idx_end or ""))

        console.print(table)
        return heading_chapters

    # Fallback: original TOC / LLM-based logic
    book = epub_load(path)
    sections = epub_spine_texts(book)

    console.rule("[bold]Stage 1 (EPUB): Try EPUB nav/TOC[/bold]")
    outline = epub_try_outline(book)
    if outline:
        console.print(f"[green]Found EPUB TOC with {len(outline)} entries.[/green]")
        chapters = []
        for (start, end, title) in outline:
            text = epub_slice_sections(sections, start, end)
            chapters.append(Chapter(idx_start=start, idx_end=end, title=title, text=text))
        return chapters

    console.print("[yellow]No usable EPUB TOC. Using LLM to infer chapters.[/yellow]")
    console.rule("[bold]Stage 2 (EPUB): LLM TOC inference from first sections[/bold]")
    toc_guess = ask_llm_for_toc(sections[:lookahead_sections], model=model, mode="epub")

    if not toc_guess:
        toc_guess = manual_toc if manual_toc else request_manual_toc_from_stdin()

    console.rule("[bold]Stage 3 (EPUB): Align to spine sections[/bold]")
    # For EPUB we align directly to the section index (no page headings)
    starts: List[int] = []
    titles_by_idx: Dict[int, str] = {}
    for item in toc_guess:
        ss = item.get("start_section")
        if isinstance(ss, int) and ss >= 1:
            idx = min(len(sections)-1, max(0, ss - 1))
            starts.append(idx)
            titles_by_idx[idx] = (str(item.get("title") or "").strip() or f"Chapter starting at section {idx+1}")
    if 0 not in starts:
        starts.append(0)
        titles_by_idx.setdefault(0, "Chapter starting at section 1")
    starts = sorted(set(starts))

    chapters: List[Chapter] = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else None
        title = titles_by_idx.get(s)
        if not title:
            # fallback: first non-empty line in section
            lines = [l.strip() for l in (sections[s].splitlines() if sections[s] else []) if l.strip()]
            title = (lines[0][:120] if lines else f"Chapter starting at section {s+1}")
        text = epub_slice_sections(sections, s, e)
        chapters.append(Chapter(idx_start=s, idx_end=e, title=title, text=text))

    # Show alignment
    table = Table(title="Aligned Chapters (EPUB)", show_lines=True)
    table.add_column("#", justify="right"); table.add_column("Title")
    table.add_column("Start section (idx+1)"); table.add_column("End section idx")
    for i, ch in enumerate(chapters, start=1):
        table.add_row(str(i), ch.title, str(ch.idx_start+1), str(ch.idx_end or len(sections)))
    console.print(table)
    return chapters

# ---------------- Orchestration ----------------
def write_markdown(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def main(book_path: str, out_dir: str, model: str, lookahead: int,
         force: bool, manual_toc_file: Optional[str], epub_page_size: int):
    console.rule(f"[bold]Start[/bold]  Book: {book_path}")
    console.print(f"Ollama host: [cyan]{OLLAMA_HOST}[/cyan]  |  Model: [magenta]{model}[/magenta]")
    console.print(f"Output dir: [green]{out_dir}[/green]  |  TOC scan window: {lookahead}")

    manual_toc: Optional[List[Dict[str, Any]]] = None
    if manual_toc_file:
        console.print(Panel.fit(f"Loading manual TOC file: {manual_toc_file}", style="bold yellow"))
        manual_toc = load_manual_toc_file(manual_toc_file)

    ext = pathlib.Path(book_path).suffix.lower()
    if ext == ".pdf":
        with console.status("[bold cyan]Building chapters (PDF)…[/bold cyan]"):
            chapters = build_chapters_pdf(book_path, lookahead, model, manual_toc)
    elif ext == ".epub":
        console.print(f"EPUB synthetic page size: {epub_page_size} chars")
        with console.status("[bold cyan]Building chapters (EPUB)…[/bold cyan]"):
            chapters = build_chapters_epub(book_path, lookahead, model, manual_toc, epub_page_size)
    else:
        console.print(f"[red]Unsupported file extension: {ext}. Use PDF or EPUB.[/red]")
        sys.exit(1)

    console.rule("[bold]Stage 4: Summarization (thorough, linear)[/bold]")
    summary_dir = pathlib.Path(out_dir)
    index_rows = []

    for i, ch in enumerate(chapters, start=1):
        safe_name = re.sub(r"[^-\w]+", "_", ch.title).strip("_") or f"chapter_{i}"
        out_md = summary_dir / f"{i:02d}_{safe_name}.md"

        if out_md.exists() and not force:
            console.print(f"[dim]Skipping (exists): {out_md.name}[/dim]")
            index_rows.append((i, ch.title, str(out_md)))
            continue

        # Log chapter info
        rng = f"{ch.idx_start+1} → {(ch.idx_end or 'EOF')}"
        console.print(Panel.fit(f"[bold]Chapter {i}[/bold]: {ch.title}\nSpan: {rng}", style="bold blue"))

        # Chunk and summarize
        chunks = chunk_text(ch.text, MAX_CHUNK_CHARS, CHUNK_OVERLAP)
        console.print(f"Chunking → {len(chunks)} chunk(s) (≈{len(ch.text)} chars)")

        chunk_summaries = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            t = progress.add_task(f"Summarizing chapter {i}", total=len(chunks))
            for j, chunk in enumerate(chunks, start=1):
                console.print(f"[cyan]→ LLM call[/cyan] Chapter {i} – chunk {j}/{len(chunks)} (len={len(chunk)})")
                try:
                    cs = summarize_chunk(ch.title, chunk, model)
                except requests.HTTPError:
                    time.sleep(2)
                    cs = summarize_chunk(ch.title, chunk, model)
                chunk_summaries.append(cs)
                progress.advance(t)

        final_summary = chunk_summaries[0] if len(chunk_summaries) == 1 else merge_chunk_summaries(ch.title, chunk_summaries, model)

        final_doc = f"# {ch.title}\n\n> Chapter span index start: {ch.idx_start+1}\n\n{final_summary}\n"
        write_markdown(out_md, final_doc)
        console.print(f"[green]Wrote[/green] {out_md.name}")
        index_rows.append((i, ch.title, str(out_md)))

    # Write index
    index_md = summary_dir / "_index.md"
    lines = ["# Chapter Summaries Index\n"]
    for i, title, path in index_rows:
        lines.append(f"- **Chapter {i}** — {title} → `{pathlib.Path(path).name}`")
    write_markdown(index_md, "\n".join(lines) + "\n")
    console.rule("[bold green]Done[/bold green]")
    console.print(f"Chapters summarized: {len(index_rows)}")
    console.print(f"Index: {index_md}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM-driven chapter detection + thorough summaries for PDF/EPUB (Ollama).")
    ap.add_argument("book", help="Path to the book (PDF or EPUB)")
    ap.add_argument("--out", default=OUT_DIR, help="Directory to write markdown summaries")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name, e.g., gpt-oss:20b")
    ap.add_argument("--pages", type=int, default=20, help="How many initial units to show the LLM (PDF pages or EPUB sections)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing summaries")
    ap.add_argument("--manual-toc", help="Path to a JSON file with TOC (PDF: title+start_page; EPUB: title+start_section)")
    ap.add_argument(
        "--epub-page-size",
        type=int,
        default=EPUB_PAGE_SIZE_DEFAULT,
        help="Characters per synthetic EPUB page when using heading-based indexing",
    )
    args = ap.parse_args()
    main(args.book, args.out, args.model, args.pages, args.force, args.manual_toc, args.epub_page_size)
