"""The thin index.html of a profile dump: a run summary and verdict, a kernel table (cycles, speed of light,
verdict, links to tile detail, source and dump files), and links to trace.pftrace (Open in Perfetto),
TORCH_TRACE and tlparse. No charts: the analysis happens in Perfetto, the MLIR files and tlparse."""
import glob
import html
import os
import sys
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_sol  # noqa: E402
import profile_trace  # noqa: E402

_STAGE_SUFFIXES = (".mlir", ".s")

_STYLE = """
:root { --bg: #f9f9f7; --card: #fcfcfb; --ink: #0b0b0b; --muted: #6b6a66; --line: #e1e0d9; --link: #2a78d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d0d0d; --card: #1a1a19; --ink: #f2f2f0; --muted: #a3a29c;
          --line: #2c2c2a; --link: #6aa5ee; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 13px; margin: 24px 0 8px; color: var(--muted); text-transform: uppercase;
     letter-spacing: .04em; }
p { margin: 0 0 8px; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: var(--muted); }
.scroll { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.links a { margin-right: 10px; white-space: nowrap; display: inline-block; }
button { font: inherit; padding: 4px 12px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--card); color: var(--ink); cursor: pointer; }
button:hover { border-color: var(--link); color: var(--link); }
ul { margin: 0; padding-left: 18px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; white-space: nowrap;
         border: 1px solid var(--line); }
.v-compute { background: #dcebfb; color: #174a85; border-color: #a9c9ee; }
.v-memory { background: #fbe3d0; color: #8a3d0a; border-color: #f0bc98; }
.v-dma { background: #f5e0f3; color: #7a2a73; border-color: #e2b4dd; }
.v-latency { background: #e7e7e3; color: #4b4a46; }
.v-unknown { color: var(--muted); }
@media (prefers-color-scheme: dark) {
  .v-compute { background: #12304f; color: #a9cdf5; border-color: #24517f; }
  .v-memory { background: #4a2a12; color: #f3c9a3; border-color: #7a4a24; }
  .v-dma { background: #432040; color: #eec3ea; border-color: #6e3a69; }
  .v-latency { background: #2e2e2c; color: #c4c3bd; }
}
"""

_SCRIPT = """
const PERFETTO = 'https://ui.perfetto.dev';
const cache = {};
let pending = null;
const note = (t) => { document.getElementById('note').textContent = t; };
function post(file, buffer) {
  const win = window.open(PERFETTO);
  if (!win) { note('The browser blocked the new tab.'); return; }
  const ping = setInterval(() => win.postMessage('PING', PERFETTO), 50);
  window.addEventListener('message', function onPong(e) {
    if (e.source !== win || e.data !== 'PONG') return;
    clearInterval(ping);
    window.removeEventListener('message', onPong);
    win.postMessage({perfetto: {buffer: buffer, title: file}}, PERFETTO);
  });
}
async function openInPerfetto(file) {
  if (!cache[file]) {
    try { cache[file] = await (await fetch(file)).arrayBuffer(); }
    catch (e) {
      pending = file;
      document.getElementById('pick').hidden = false;
      note('Cannot read ' + file + ' from here: choose it below.');
      return;
    }
  }
  post(file, cache[file]);
}
document.querySelectorAll('.open').forEach((b) =>
  b.addEventListener('click', () => openInPerfetto(b.dataset.file)));
document.getElementById('pick').addEventListener('change', async (e) => {
  cache[pending] = await e.target.files[0].arrayBuffer();
  note('Loaded; press the button again.');
});
"""


def get_link(path, out):
    """`path` as a URL relative to the directory `out`, or None when it does not exist."""
    rel = profile_trace.get_rel_path(path, out)
    return quote(rel, safe="/") if rel else None


def find_timing(roots):
    """workdir -> its timing.json for every timing.json under the roots (dirs or files)."""
    out = {}
    for root in roots:
        if os.path.isdir(root):
            files = glob.glob(os.path.join(root, "**", "timing.json"), recursive=True)
        else:
            files = [root] if os.path.basename(root) == "timing.json" else []
        out.update({os.path.dirname(os.path.abspath(f)): f for f in files})
    return out


def get_stage_files(workdir):
    """The numbered stage outputs (01-ttir.mlir ... 11-kernel.s) in `workdir`, earliest first."""
    names = sorted(n for n in os.listdir(workdir) if n[:2].isdigit() and n[2:3] == "-"
                   and n.endswith(_STAGE_SUFFIXES)) if os.path.isdir(workdir) else []
    return [os.path.join(workdir, n) for n in names]


def get_source_file(workdir):
    """The generated Triton source in `workdir`, or None."""
    files = sorted(glob.glob(os.path.join(workdir, "*_triton.py")))
    return files[0] if files else None


def summarize_kernels(live, kernel_runs, tiles, timing):
    """One row per distinct kernel workdir, biggest first: launches, cycles including copy_node runs, the
    share of all run cycles, the speed of light merged over its launches and the files worth linking."""
    total = sum(r["cycles"] for r in live) or 1
    rows = {}
    for kernels in kernel_runs:
        for k in kernels:
            if k["copy"]:
                continue
            row = rows.setdefault(k["workdir"], {"short": k["short"], "dir": k["dir"],
                                                 "workdir": k["workdir"], "launches": 0, "cycles": 0,
                                                 "sols": []})
            row["launches"] += 1
            row["cycles"] += k["end"] - k["start"] + k["copy_cycles"]
            row["sols"].append((k["span"][1] - k["span"][0], k["sol"]))
    out = sorted(rows.values(), key=lambda r: -r["cycles"])
    for row in out:
        row["share"] = 100.0 * row["cycles"] / total
        row["tile"] = tiles.get(row["dir"])
        own = os.path.join(row["workdir"], "timing.json")
        row["timing"] = timing.get(row["workdir"]) or (own if os.path.exists(own) else None)
        row["stages"] = get_stage_files(row["workdir"])
        row["source"] = get_source_file(row["workdir"])
        row["sol"] = profile_sol.merge_sols(row.pop("sols"))
    return out


def _render_sol_cells(sol):
    """The compute % (unit), DRAM %, DMA % and verdict table cells of a speed-of-light dict."""
    def pct(v):
        return "-" if v is None else f"{v:.1f}%"
    unit = f" {html.escape(sol['compute_unit'])}" if sol["compute_unit"] else ""
    verdict = html.escape(sol["verdict"])
    badge = f'<span class="badge v-{profile_sol.get_verdict_class(sol["verdict"])}">{verdict}</span>'
    return (f'<td class="n">{pct(sol["compute_pct"])}{unit}</td><td class="n">{pct(sol["dram_pct"])}</td>'
            f'<td class="n">{pct(sol["dma_pct"])}</td><td>{badge}</td>')


def _render_run_verdict(rows):
    """One line: the cycle-weighted verdict of all kernels and the share of cycles per verdict."""
    sol = profile_sol.merge_sols([(r["cycles"], r["sol"]) for r in rows])
    by = {}
    for r in rows:
        by[r["sol"]["verdict"]] = by.get(r["sol"]["verdict"], 0) + r["cycles"]
    total = sum(by.values()) or 1
    ranked = sorted(by.items(), key=lambda x: -x[1])
    split = ", ".join(f"{100.0 * c / total:.0f}% {html.escape(v)}" for v, c in ranked)
    nums = "; ".join(x for x in (
        f"compute {sol['compute_pct']:.1f}% {sol['compute_unit']}" if sol["compute_pct"] is not None else "",
        f"DRAM {sol['dram_pct']:.1f}% of peak" if sol["dram_pct"] is not None else "",
        f"DMA {sol['dma_pct']:.1f}% busy" if sol["dma_pct"] is not None else "") if x)
    return (f"Run verdict: <b>{html.escape(sol['verdict'])}</b> ({nums}; cycle-weighted over kernels). "
            f"By cycles: {split}.")


def _render_links(row, out):
    """The row's links, tile detail (with its Perfetto button), source and the earliest stage first."""
    first = [("tile detail", row["tile"] and quote(row["tile"], safe="/")),
             ("source", get_link(row["source"], out))]
    first += [(os.path.basename(f), get_link(f, out)) for f in row["stages"][:1]]
    rest = [("dump", get_link(row["workdir"], out))]
    rest += [(os.path.basename(f), get_link(f, out)) for f in row["stages"][1:]]
    rest += [("timing.json", get_link(row["timing"], out))]
    cells = "".join(f'<a href="{href}">{html.escape(label)}</a>' for label, href in first + rest if href)
    if row["tile"]:
        cells = (f'<button class="open" data-file="{quote(row["tile"], safe="/")}">tile in Perfetto</button> '
                 + cells)
    return cells


def render_index(out, live, empty, rows, extra_links, trace_name="trace.pftrace"):
    """The index.html text: summary line, run verdict, kernel table and links; paths are relative to `out`."""
    total = sum(r["cycles"] for r in live)
    machine = next((r["machine"] for r in live if r["machine"]), "unknown")
    mhz = next((r["config"]["core_freq_mhz"] for r in live if r["config"].get("core_freq_mhz")), 1000)
    launches = sum(r["launches"] for r in rows)
    body = ["<h1>NPU profile</h1>",
            f'<p class="muted">{total:,} cycles &middot; {launches} kernel launches ({len(rows)} distinct) '
            f"&middot; {len(live)} runs, {len(empty)} empty &middot; machine {html.escape(machine)}</p>",
            f"<p>{_render_run_verdict(rows)}</p>" if rows else "",
            "<h2>Trace</h2>",
            f'<p><a href="{trace_name}">{trace_name}</a> &middot; '
            f'<button id="open" class="open" data-file="{trace_name}">Open in Perfetto</button> '
            '<input id="pick" type="file" hidden> <span id="note" class="muted"></span></p>',
            '<p class="muted">Time axis: the simulated runs laid end to end in log order; '
            f"1 &micro;s = {mhz} cycles (the core clock); every slice also carries its raw cycles.</p>"]
    if extra_links:
        body.append("<ul>" + "".join(f'<li><a href="{href}">{html.escape(label)}</a></li>'
                                     for label, href in extra_links) + "</ul>")
    body += ["<h2>Kernels</h2>", f'<p class="muted">{html.escape(profile_sol.get_note())}</p>',
             '<div class="scroll"><table><thead><tr><th>kernel</th>'
             '<th class="n">launches</th><th class="n">cycles (incl. copy)</th><th class="n">share</th>'
             '<th class="n">compute (unit)</th><th class="n">DRAM</th><th class="n">DMA</th>'
             "<th>verdict</th><th>files</th></tr></thead><tbody>"]
    for row in rows:
        body.append(f'<tr><td>{html.escape(row["short"])}<br><span class="muted">{html.escape(row["dir"])}'
                    f'</span></td><td class="n">{row["launches"]}</td><td class="n">{row["cycles"]:,}</td>'
                    f'<td class="n">{row["share"]:.1f}%</td>{_render_sol_cells(row["sol"])}'
                    f'<td class="links">{_render_links(row, out)}</td></tr>')
    body.append("</tbody></table></div>")
    if empty:
        body.append('<p class="muted">Empty runs (no kernel scheduled): '
                    + html.escape(", ".join(r["log"] for r in empty)) + "</p>")
    return ('<!doctype html>\n<html lang="en"><head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>NPU profile</title>\n'
            f"<style>{_STYLE}</style></head><body><main>\n" + "\n".join(body)
            + f"\n</main><script>{_SCRIPT}</script></body></html>\n")
