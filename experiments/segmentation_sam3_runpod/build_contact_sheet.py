"""Build a single HTML contact sheet of all per-image results so the
268-image set is browsable in one scroll.

For each image_id under results/, lays out:
    [input]  [ceiling_item_overlay]  [light_overlay]  [vent_overlay]
plus the per-prompt detection counts.

Output: experiments/segmentation_sam3_runpod/index.html
Open with:
    open experiments/segmentation_sam3_runpod/index.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path


PROMPTS = ["ceiling item", "light", "vent"]
PROMPT_COLORS = {
    "ceiling item": "#ff4040",
    "light": "#ffdc40",
    "vent": "#40c8ff",
}


def slug(s: str) -> str:
    return s.replace(" ", "_")


def main() -> None:
    here = Path(__file__).parent
    results_dir = here / "results"
    image_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

    rows: list[str] = []
    counts_total = {p: 0 for p in PROMPTS}
    elapsed_total = 0.0

    for img_dir in image_dirs:
        det_path = img_dir / "detections.json"
        if not det_path.exists():
            continue
        det = json.loads(det_path.read_text())
        elapsed_total += det.get("elapsed_s", 0.0)
        per_prompt: dict[str, int | str] = {}
        for p in det.get("prompts", []):
            n = p.get("count")
            if n is None and "error" in p:
                n = "ERR"
            per_prompt[p["prompt"]] = n
            if isinstance(n, int):
                counts_total[p["prompt"]] = counts_total[p["prompt"]] + n

        cells = []
        cells.append(
            f'<a href="results/{img_dir.name}/input.jpg" target="_blank">'
            f'<img src="results/{img_dir.name}/input.jpg" loading="lazy"></a>'
        )
        for prompt in PROMPTS:
            ov = f"results/{img_dir.name}/{slug(prompt)}_overlay.jpg"
            n = per_prompt.get(prompt, "—")
            color = PROMPT_COLORS[prompt]
            cells.append(
                f'<div class="cell">'
                f'<a href="{ov}" target="_blank">'
                f'<img src="{ov}" loading="lazy"></a>'
                f'<div class="badge" style="background:{color}">{prompt}: '
                f'{n}</div>'
                f'</div>'
            )
        elapsed = det.get("elapsed_s", 0.0)
        rows.append(
            f'<section class="row" id="img-{html.escape(img_dir.name)}" '
            f'data-id="{html.escape(img_dir.name)}" '
            f'data-ci="{per_prompt.get("ceiling item", 0) or 0}" '
            f'data-light="{per_prompt.get("light", 0) or 0}" '
            f'data-vent="{per_prompt.get("vent", 0) or 0}">'
            f'<header><a href="#img-{html.escape(img_dir.name)}">'
            f'<code>{html.escape(img_dir.name)}</code></a> '
            f'<span class="meta">({elapsed:.1f}s)</span></header>'
            f'<div class="grid">{cells[0]}{cells[1]}{cells[2]}{cells[3]}</div>'
            f'</section>'
        )

    body_count = len(rows)
    avg_per_prompt = {p: counts_total[p] / max(body_count, 1) for p in PROMPTS}

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAM3 (RunPod) — contact sheet</title>
<style>
  :root {{
    --bg: #0e0f12;
    --panel: #16181d;
    --text: #d8dce3;
    --muted: #8a8f99;
    --accent: #6cf;
    --gap: 12px;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 13px/1.4 -apple-system, system-ui, sans-serif;
  }}
  header.top {{
    position: sticky; top: 0; z-index: 10;
    background: var(--panel);
    padding: 12px 16px;
    border-bottom: 1px solid #222;
    display: flex; gap: 24px; align-items: baseline; flex-wrap: wrap;
  }}
  header.top h1 {{ font-size: 14px; margin: 0; }}
  header.top .stats {{ color: var(--muted); }}
  header.top .stats b {{ color: var(--text); }}
  .controls {{ margin-left: auto; display: flex; gap: 12px; }}
  .controls label {{ color: var(--muted); }}
  .controls input, .controls select {{
    background: #0a0b0d; color: var(--text);
    border: 1px solid #2a2c33; padding: 4px 8px;
    border-radius: 4px;
  }}
  main {{ padding: 16px; }}
  section.row {{
    background: var(--panel);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: var(--gap);
  }}
  section.row > header {{
    display: flex; gap: 12px; align-items: baseline;
    padding: 4px 8px 8px;
  }}
  section.row > header a {{ color: var(--accent); text-decoration: none; }}
  section.row > header .meta {{ color: var(--muted); }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--gap);
  }}
  .grid > a, .grid > .cell {{
    position: relative;
    display: block;
    background: #0a0b0d;
    border-radius: 4px;
    overflow: hidden;
    aspect-ratio: 3/4;
  }}
  .grid img {{
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
  }}
  .badge {{
    position: absolute;
    top: 6px; left: 6px;
    color: #000;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 11px;
    pointer-events: none;
  }}
  @media (max-width: 900px) {{
    .grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<header class="top">
  <h1>SAM3 (RunPod, self-hosted)</h1>
  <div class="stats">
    <b>{body_count}</b> images ·
    avg ceiling-item <b>{avg_per_prompt['ceiling item']:.1f}</b> ·
    avg light <b>{avg_per_prompt['light']:.1f}</b> ·
    avg vent <b>{avg_per_prompt['vent']:.1f}</b> ·
    total inference <b>{elapsed_total/60:.1f} min</b>
  </div>
  <div class="controls">
    <label>filter id <input id="q" placeholder="431…"></label>
    <label>min ceiling <input id="min-ci" type="number" min="0" value="0" style="width:60px"></label>
    <label>min light <input id="min-light" type="number" min="0" value="0" style="width:60px"></label>
    <label>min vent <input id="min-vent" type="number" min="0" value="0" style="width:60px"></label>
    <label>sort
      <select id="sort">
        <option value="id">by id</option>
        <option value="ci">most ceiling items</option>
        <option value="light">most lights</option>
        <option value="vent">most vents</option>
      </select>
    </label>
  </div>
</header>
<main id="rows">
{chr(10).join(rows)}
</main>
<script>
  const main = document.getElementById('rows');
  const rows = Array.from(main.querySelectorAll('section.row'));
  const q = document.getElementById('q');
  const minCi = document.getElementById('min-ci');
  const minLight = document.getElementById('min-light');
  const minVent = document.getElementById('min-vent');
  const sortSel = document.getElementById('sort');
  function apply() {{
    const qv = q.value.trim();
    const mi = +minCi.value, ml = +minLight.value, mv = +minVent.value;
    const sk = sortSel.value;
    const visible = rows.filter(r => {{
      if (qv && !r.dataset.id.includes(qv)) return false;
      if (+r.dataset.ci < mi) return false;
      if (+r.dataset.light < ml) return false;
      if (+r.dataset.vent < mv) return false;
      return true;
    }});
    if (sk !== 'id') {{
      const key = sk;
      visible.sort((a,b) => +b.dataset[key] - +a.dataset[key]);
    }} else {{
      visible.sort((a,b) => a.dataset.id.localeCompare(b.dataset.id));
    }}
    main.replaceChildren(...visible);
  }}
  [q, minCi, minLight, minVent, sortSel].forEach(el => el.addEventListener('input', apply));
</script>
</body>
</html>
"""
    out = here / "index.html"
    out.write_text(html_doc)
    print(f"wrote {out} ({body_count} images)")


if __name__ == "__main__":
    main()
