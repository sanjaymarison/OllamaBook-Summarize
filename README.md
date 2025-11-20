
# Book‑Summarizer

A single‑file Python tool that:

* Detects chapters in a PDF or EPUB
* Breaks each chapter into manageable chunks
* Summarises each chunk with an Ollama‑backed LLM
* Merges the chunk summaries into a single, linear, paragraph‑only chapter summary
* Saves every chapter as a separate Markdown file
* Generates a tiny `_index.md` that links to all chapter files

All of this is driven entirely from the command line – no GUI, no configuration files, no complex build steps.

---

## 1.  What it can do

| Feature | What it actually does |
|---------|-----------------------|
| **Chapter detection** | • Uses the PDF outline if present. <br>• Uses the EPUB nav/TOC if present. <br>• If no outline, asks an LLM to infer chapter boundaries from the first N pages/sections. <br>• If that fails, lets the user paste a manual TOC in JSON. |
| **Chunking** | Splits a chapter into pieces of ≤ 12 k characters with up to 800 characters overlap. Dedupes near‑duplicate chunks. |
| **LLM summarisation** | Calls the Ollama REST API (`/api/generate`). <br>• Uses the *gpt‑oss:20b* default model or a model you specify. <br>• Produces a **thorough, linear** paragraph‑only summary – no quotes, no bullets, no Q&A. |
| **Output** | • One Markdown file per chapter (`01_Chapter_Title.md`). <br>• An index file (`_index.md`) that lists all chapters with relative links. |
| **Overwrite control** | Skips existing files unless `--force` is used. |
| **Manual TOC** | Accepts a JSON file (or STDIN) that lists `{title, start_page}` for PDF or `{title, start_section}` for EPUB. |

---

## 2.  Prerequisites

| Dependency | Why it’s needed |
|------------|-----------------|
| **Python 3.9+** | The code uses modern type hints and f‑strings. |
| `pdfplumber` | Extract text from PDF pages. |
| `pypdf` | Read PDF outline (if present). |
| `ebooklib` | Read EPUB files and nav/TOC. |
| `beautifulsoup4` | Parse internal EPUB XHTML for headings and “Contents” pages. |
| `tqdm` | Progress bars for long LLM calls. |
| `rich` | Pretty console output (tables, panels, progress). |
| `requests` | HTTP client for Ollama API. |
| `python‑m‑pip install -r requirements.txt` | Optional – you can simply install the packages listed above. |

> **Ollama requirement** – The code talks to an Ollama server over HTTP.  
> Set the server URL with the environment variable `OLLAMA_HOST` (default: `http://localhost:11434`).

---

## 3.  Quick start

```bash
# Install dependencies (optional but recommended)
pip install pdfplumber pypdf ebooklib beautifulsoup4 tqdm rich requests

# Run the summariser
python book_summariser.py path/to/book.pdf            # or .epub
# Optional arguments
#   --out OUT_DIR          # Where Markdown files are written (default: "summaries")
#   --model MODEL          # Ollama model name (default: gpt-oss:20b)
#   --pages N              # Number of initial pages/sections to feed to LLM for TOC inference (default: 20)
#   --force                # Overwrite existing chapter files
#   --manual-toc FILE      # JSON file with manual TOC if automatic detection fails
```

Example:

```bash
python book_summariser.py mybook.epub --model llama3.2:13b --pages 30
```

The script will print progress to the console, then drop files into `summaries/`.

---

## 4.  Manual TOC format

If the book has no usable outline, you can supply a JSON file:

```json
[
  {"title": "Chapter 1: Introduction", "start_page": 5},
  {"title": "Chapter 2: Methods",       "start_page": 23}
]
```

* For PDFs the key is `start_page` (1‑based as shown in a PDF viewer).  
* For EPUBs the key is `start_section` (1‑based by spine order).  
* Every chapter must be in reading order.

You can also paste the JSON directly into the terminal when prompted.

---

## 5.  What the summarisation looks like

Each chapter file begins with:

```markdown
# Chapter Title

> Chapter span index start: 12

<the paragraph‑only summary goes here…>
```

The summary is a **single, continuous paragraph** that follows the book’s flow.  
It contains no quotes, bullet points, questions, or meta‑commentary.

The index file `_index.md` simply lists each chapter:

```markdown
# Chapter Summaries Index

- **Chapter 1** — Introduction → `01_Introduction.md`
- **Chapter 2** — Methods        → `02_Methods.md`
```

---

## 6.  Limitations & gotchas

| Limitation | Why it matters |
|------------|----------------|
| **Only PDF/EPUB** | The script never touches other formats. |
| **Ollama dependency** | If the Ollama server is unreachable or returns bad JSON, chapter detection may fall back to manual input or fail silently. |
| **LLM call failures** | On HTTP errors the code retries once after a short pause; persistent failures abort the chunk’s summarisation. |
| **Chunk size** | 12 k chars is a hard limit; very long chapters may still be broken into many chunks, increasing API calls. |
| **Overlap** | 800‑char overlap prevents context loss but may still miss subtle connections across chunk boundaries. |
| **No citations or metadata** | The summary contains no links back to the original text or source attribution. |
| **No formatting preservation** | Summaries are plain text; formatting like tables, images, or footnotes is lost. |
| **Manual TOC required for some books** | If the book has no outline and the LLM cannot infer chapters, you must supply a manual TOC. |
| **No user‑defined summarisation style** | The style is hard‑coded to “thorough, linear”. |
| **Output is written to disk only** | No API or in‑memory output; you have to read the Markdown files. |

---

## 7.  Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `requests.exceptions.ConnectionError` | Ollama host unreachable | Check `OLLAMA_HOST` and ensure the Ollama server is running and reachable. |
| LLM returns garbage or empty summary | Model not available or too large | Switch to a smaller or available model (`--model llama3.1:8b`). |
| Chapter list shows wrong start pages | PDF/EPUB has no outline, LLM failed | Provide a manual TOC (`--manual-toc`). |
| Script stalls on “Querying LLM to infer chapters” | Slow network or large prompt | Increase `--pages` to give more context, or reduce to a smaller number if you’re hitting token limits. |
| Output files not created | Script stopped with an exception | Look at the console output for stack traces; often a missing dependency or a bad URL. |

---

## 8.  Customising the script

If you want to tweak any behaviour (e.g., chunk size, overlap, or the LLM prompt), open the source file and edit the constants near the top:

```python
MAX_CHUNK_CHARS = 12000   # chunk size in characters
CHUNK_OVERLAP   = 800     # overlap between chunks
SYSTEM_SUMMARY  = """You are a careful book summarizer…
```

The prompt strings (`SYSTEM_SUMMARY`, `SYSTEM_TOC_PDF`, `SYSTEM_TOC_EPUB`) are where you would change the summarisation style or TOC inference rules.

---

## 9.  Credits

* **Ollama** – The LLM provider.  
* **pdfplumber, pypdf, ebooklib, beautifulsoup4** – Text extraction libraries.  
* **rich** – Console UI.  
* **tqdm** – Progress bars.  
* **requests** – HTTP client.

---

## 10.  License

This script is released under the MIT License. Feel free to fork, extend, or use it in your own projects.
