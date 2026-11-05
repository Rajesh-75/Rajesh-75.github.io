"""
unify_styles.py  -  build styles_common.css from ONE reference page and link it
into every page, removing only the page styles that are now redundant.

Safe by design:
  * DRY_RUN = True  -> nothing is written except the report. Read it first.
  * Originals are backed up before any page is touched.
  * A page rule is removed ONLY if the common CSS already has the same
    selector (same @media context) with IDENTICAL declarations.
  * Everything else (unique rules, and rules that differ from common = page
    overrides) stays in the page's own <style> block, after the common link,
    so it still wins the cascade.
  * Identical rules shared by >= PROMOTE_MIN_PAGES pages (and not already in
    common) are promoted into styles_common.css.

Run from IDLE: edit the settings below, press F5.
"""
import re
import shutil
from pathlib import Path
from collections import defaultdict

# ----------------------------- SETTINGS ------------------------------------
SITE_DIR = Path(r"D:\Rajesh\Work\Stylecorrectpages\WebPageswithcommonStyles")      # folder containing the .html pages
REFERENCE_PAGE = "knn.html"    # page whose look you want
COMMON_NAME = "styles_common.css"
PROMOTE_MIN_PAGES = 2                         # promote shared unique rules; 0 = never
DRY_RUN = False                                # set False to actually modify pages
SKIP_PAGES = set()                            # e.g. {"index.html"}
MASTER_CSS = "style.css"                      # existing sheet: only checked for collisions, never edited
FORCE_COMMON_SELECTORS = set()                # e.g. {"body","h1","h2","h3","p","li","a","code"}: for these
                                              # selectors the COMMON style wins; the page's differing
                                              # declarations are removed (listed in the report)
COPY_FONT_LINKS = True                        # copy the reference page's Google Fonts <link> tags into every page
# ---------------------------------------------------------------------------

# Only plain <style> blocks are touched. <style media="print"> etc. are left alone,
# and so is any <style> that sits inside an inline <svg>.
STYLE_RE = re.compile(r"(<style>)(.*?)(</style>)", re.S | re.I)
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S | re.I)


def head_style_matches(text):
    spans = [m.span() for m in SVG_RE.finditer(text)]
    return [m for m in STYLE_RE.finditer(text)
            if not any(a <= m.start() < b for a, b in spans)]
LINK_TAG = f'<link rel="stylesheet" href="{COMMON_NAME}">'


def parse(css, ctx=()):
    """Return list of (ctx_tuple, head, decls_tuple_or_None). Comments dropped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out, i, n = [], 0, len(css)
    while i < n:
        j, s = css.find("{", i), css.find(";", i)
        if s != -1 and (j == -1 or s < j):          # e.g. @import ...;
            stmt = css[i:s + 1].strip()
            if stmt:
                out.append((ctx, stmt, None))
            i = s + 1
            continue
        if j == -1:
            break
        head = re.sub(r"\s+", " ", css[i:j].strip())
        depth, k = 1, j + 1
        while k < n and depth:
            depth += (css[k] == "{") - (css[k] == "}")
            k += 1
        body = css[j + 1:k - 1]
        if re.match(r"@(media|supports|container|layer)\b", head, re.I):
            out += parse(body, ctx + (head,))
        elif head.startswith("@"):                   # @font-face, @keyframes...
            out.append((ctx, head, (re.sub(r"\s+", " ", body.strip()),)))
        else:
            decls = tuple(
                re.sub(r"\s+", " ", d.strip())
                for d in re.split(r";(?![^(]*\))", body) if d.strip())
            out.append((ctx, head, decls))
        i = k
    return out


def key_of(rule):
    ctx, head, decls = rule
    if head.startswith("@"):                         # at-rules: whole text identity
        return (ctx, head + "|" + repr(decls))
    return (ctx, head)


def serialize(rules):
    """Pretty-print rules, re-grouping consecutive rules that share a context."""
    lines, open_ctx = [], ()
    for ctx, head, decls in rules:
        common = 0
        while common < min(len(open_ctx), len(ctx)) and open_ctx[common] == ctx[common]:
            common += 1
        for _ in range(len(open_ctx) - common):
            lines.append("}")
        for c in ctx[common:]:
            lines.append(c + " {")
        open_ctx = ctx
        pad = "  " * len(ctx)
        if decls is None:
            lines.append(pad + head)
        elif head.startswith("@"):
            lines.append(f"{pad}{head} {{ {decls[0]} }}")
        else:
            lines.append(f"{pad}{head} {{")
            lines += [f"{pad}  {d};" for d in decls]
            lines.append(pad + "}")
    lines += ["}"] * len(open_ctx)
    return "\n".join(lines) + "\n"


def is_anti_flicker(rule):
    """body{visibility:hidden} is the MathJax anti-flicker hack. It must stay in
    each page (a page without the MathJax pageReady hook would stay blank)."""
    ctx, head, decls = rule
    return (not ctx and head == "body" and decls is not None
            and any(re.sub(r"\s", "", d).lower() == "visibility:hidden" for d in decls))


FONT_LINK_RE = re.compile(
    r'<link\b[^>]*(?:rel=["\']preconnect["\']|fonts\.googleapis\.com)[^>]*>', re.I)


def page_rules(html):
    rules = []
    for m in head_style_matches(html):
        rules += parse(m.group(2))
    return rules


def main():
    pages = sorted(p for p in SITE_DIR.glob("*.html")
                   if p.name not in SKIP_PAGES)
    ref = SITE_DIR / REFERENCE_PAGE
    if not ref.exists():
        raise SystemExit(f"Reference page not found: {ref}")

    html = {p: p.read_text(encoding="utf-8") for p in pages}
    if COMMON_NAME in html[ref]:
        raise SystemExit(
            "The reference page already links " + COMMON_NAME + ", so it has been processed.\n"
            "Restore the original pages from _backup_before_styles first, otherwise "
            "styles_common.css would be rebuilt from an empty page.")
    common = [r for r in page_rules(html[ref]) if not is_anti_flicker(r)]   # 1. reference styles
    font_links = FONT_LINK_RE.findall(html[ref]) if COPY_FONT_LINKS else []
    common_bodies = defaultdict(set)                 # key -> set of declaration tuples
    for r in common:
        common_bodies[key_of(r)].add(r[2])

    # 2. look for unique rules repeated identically across other pages
    seen = defaultdict(lambda: defaultdict(set))     # key -> decls -> {pages}
    for p in pages:
        if p == ref:
            continue
        for r in page_rules(html[p]):
            if key_of(r) not in common_bodies:
                seen[key_of(r)][r[2]].add(p.name)
    promoted = []
    if PROMOTE_MIN_PAGES:
        for p in pages:
            for r in page_rules(html[p]):
                k = key_of(r)
                if (k not in common_bodies and r not in promoted
                        and not is_anti_flicker(r)
                        and len(seen[k][r[2]]) >= PROMOTE_MIN_PAGES):
                    promoted.append(r)
        for r in promoted:
            common_bodies[key_of(r)].add(r[2])
        common += promoted

    # 3. per-page: split into redundant / override / unique
    report = [f"Reference: {REFERENCE_PAGE}",
              f"Common rules: {len(common)} ({len(promoted)} promoted from other pages)\n"]
    master = SITE_DIR / MASTER_CSS
    if MASTER_CSS and master.exists():
        clash = sorted({r[1] for r in parse(master.read_text(encoding="utf-8"))
                        if key_of(r) in common_bodies and r[2] not in common_bodies[key_of(r)]})
        report.append(f"{MASTER_CSS} selectors that ALSO exist in common with DIFFERENT styling "
                      f"(common loads after it and will win; check the markup of pages using them):")
        report.append("   " + ("; ".join(clash) if clash else "none") + "\n")
    new_html = {}
    for p in pages:
        rules = page_rules(html[p])
        keep, redundant, overrides, forced = [], 0, [], []
        for r in rules:
            k = key_of(r)
            if k in common_bodies and r[2] in common_bodies[k]:
                redundant += 1
            elif (k in common_bodies and r[1] in FORCE_COMMON_SELECTORS
                  and not is_anti_flicker(r)):
                redundant += 1
                forced.append(f"{r[1]} {{ {'; '.join(r[2])} }}")
            else:
                keep.append(r)
                if k in common_bodies and not is_anti_flicker(r):
                    overrides.append(r[1])
        unique = [r[1] for r in keep
                  if key_of(r) not in common_bodies and not is_anti_flicker(r)]
        report.append(f"== {p.name}: {redundant} redundant removed, "
                      f"{len(unique)} unique kept, {len(overrides)} overrides kept")
        if overrides:
            report.append("   differs from common (page override kept): "
                          + "; ".join(sorted(set(overrides))))
        if forced:
            report.append("   FORCED to common (page values dropped): " + " | ".join(forced))
        if unique:
            report.append("   unique to this page: " + "; ".join(unique))
        extra = re.findall(r'<link[^>]+rel=["\']stylesheet["\'][^>]*>', html[p], re.I)
        extra = [e for e in extra if COMMON_NAME not in e]
        if extra:
            report.append("   existing stylesheet links left untouched: " + " | ".join(extra))
        n_inline = len(re.findall(r'\sstyle\s*=\s*"', html[p]))
        if n_inline:
            report.append(f"   note: {n_inline} inline style= attributes (not touched)")

        # rebuild the page
        text, first = html[p], True

        def repl(m):
            nonlocal first
            if first:
                first = False
                body = serialize(keep)
                hrefs = re.findall(r'href=["\']([^"\']+)["\']', html[p])
                fonts = "".join(l + "\n" for l in font_links
                                if re.search(r'href=["\']([^"\']+)', l).group(1) not in hrefs)
                link = fonts + LINK_TAG + "\n" if COMMON_NAME not in html[p] else ""
                return link + (f"{m.group(1)}\n{body}{m.group(3)}" if keep else "")
            return ""                                # later <style> blocks merged into the first
        ms = head_style_matches(text)
        if ms:
            out, last = [], 0
            for m in ms:
                out += [text[last:m.start()], repl(m)]
                last = m.end()
            text = "".join(out) + text[last:]
        elif COMMON_NAME not in text:
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', text)
            fonts = "".join(l + "\n" for l in font_links
                            if re.search(r'href=["\']([^"\']+)', l).group(1) not in hrefs)
            text = re.sub(r"</head>", lambda m: fonts + LINK_TAG + "\n</head>", text, count=1, flags=re.I)
        new_html[p] = text

    (SITE_DIR / "styles_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    if DRY_RUN:
        print("\nDRY RUN - nothing modified. Review styles_report.txt, then set DRY_RUN = False.")
        return
    backup = SITE_DIR / "_backup_before_styles"
    backup.mkdir(exist_ok=True)
    for p in pages:
        if not (backup / p.name).exists():          # never overwrite an earlier backup
            shutil.copy2(p, backup / p.name)
    (SITE_DIR / COMMON_NAME).write_text(serialize(common), encoding="utf-8")
    for p, t in new_html.items():
        p.write_text(t, encoding="utf-8")
    print(f"\nDone. {COMMON_NAME} written, {len(pages)} pages updated, backups in {backup}")


if __name__ == "__main__":
    main()
