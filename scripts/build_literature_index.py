#!/usr/bin/env python3
from pathlib import Path
import html

# Run this from the repo root:  python scripts/build_literature_index.py

REPO_ROOT = Path(__file__).resolve().parents[1]
LIT_DIR = REPO_ROOT / "csj_literature"
OUTPUT_FILE = REPO_ROOT / "csj_literature.md"

def pretty_title(path: Path) -> str:
    """
    Very simple heuristic: strip extension, replace underscores and dashes with spaces.
    You can refine this later if you want nicer titles.
    """
    stem = path.stem
    title = stem.replace("_", " ").replace("-", " ")
    return title.strip()

def main():
    if not LIT_DIR.exists():
        raise SystemExit(f"Directory not found: {LIT_DIR}")

    files = sorted(
        [p for p in LIT_DIR.iterdir() if p.is_file()],
        key=lambda p: p.name.lower()
    )

    lines = []
    # Jekyll front matter
    lines.append("---")
    lines.append("layout: page")
    lines.append("title: Galileo Literature Archive")
    lines.append("permalink: /csj_literature/")
    lines.append("---")
    lines.append("")
    lines.append("# Galileo Literature Archive")
    lines.append("")
    lines.append("Type in the box below to filter the list by author, year, or title substring.")
    lines.append("")
    # Search box
    lines.append('<input id="lit-search" type="text" placeholder="Search literature..." style="width:100%; padding:0.5rem; margin-bottom:1rem;">')
    lines.append("")
    lines.append('<ul id="lit-list">')

    for f in files:
        rel_url = f"/csj_literature/{f.name}"
        title = pretty_title(f)
        data_title = html.escape(title.lower())
        display_title = html.escape(title)
        display_file = html.escape(f.name)
        lines.append(
            f'  <li data-title="{data_title}">'
            f'<a href="{rel_url}">{display_title}</a> '
            f'<span style="font-size:0.85em; color:#666;">({display_file})</span>'
            f"</li>"
        )

    lines.append("</ul>")
    lines.append("")
    # Tiny client-side search script
    lines.append("<script>")
    lines.append("  (function() {")
    lines.append("    var input = document.getElementById('lit-search');")
    lines.append("    if (!input) return;")
    lines.append("    var items = Array.prototype.slice.call(document.querySelectorAll('#lit-list li'));")
    lines.append("    input.addEventListener('input', function(e) {")
    lines.append("      var q = e.target.value.toLowerCase();")
    lines.append("      items.forEach(function(li) {")
    lines.append("        var t = li.getAttribute('data-title') || '';")
    lines.append("        li.style.display = t.indexOf(q) !== -1 ? '' : 'none';")
    lines.append("      });")
    lines.append("    });")
    lines.append("  })();")
    lines.append("</script>")
    lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE.relative_to(REPO_ROOT)} with {len(files)} items.")

if __name__ == "__main__":
    main()
