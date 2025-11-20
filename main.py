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
