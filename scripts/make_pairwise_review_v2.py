"""Reproduce the pairwise human-validation tool, CORRECTED:
 - pairs are drawn from the pairs the committee ACTUALLY judged (runs/pairwise_comparisons.parquet),
   so human choices are directly comparable to committee votes (fixes the earlier RNG mismatch);
 - stratified across committee BT-gap bands (near-tie / moderate / clear) so agreement can be
   measured as a function of how different the two papers are;
 - resume-safe UI (verified 9/9 deterministic + 200/200 swarm), storage key v3 (won't collide with
   the completed v2 ratings).
Output: reports/pairwise_review_v2.html + reports/pairwise_review_v2_items.csv
"""
from __future__ import annotations
import json, random
import pandas as pd
from sswr_eval import config, corpus

PER_BAND = 40
rng = random.Random(20260701)

comps = pd.read_parquet(config.RUNS_DIR / "pairwise_comparisons.parquet")
comps["a"] = comps[["doc_a", "doc_b"]].min(axis=1).astype(int)
comps["b"] = comps[["doc_a", "doc_b"]].max(axis=1).astype(int)
pairs = comps[["query_id", "a", "b"]].drop_duplicates()

qrels = pd.read_parquet(config.RUNS_DIR / "pairwise_qrels.parquet")
rel = {(r.query_id, int(r.paper_id)): float(r.relevance) for r in qrels.itertuples()}

rows = []
for r in pairs.itertuples():
    ra, rb = rel.get((r.query_id, r.a)), rel.get((r.query_id, r.b))
    if ra is None or rb is None:
        continue
    rows.append({"qid": r.query_id, "a": r.a, "b": r.b, "gap": abs(ra - rb)})
pf = pd.DataFrame(rows)
bands = {"near-tie (<.15)": pf[pf.gap < .15], "moderate (.15-.35)": pf[(pf.gap >= .15) & (pf.gap < .35)],
         "clear (>.35)": pf[pf.gap >= .35]}

df = corpus.load_corpus().set_index("id")
queries = pd.read_parquet(config.RUNS_DIR / "eval_queries.parquet")
qmap = dict(zip(queries["query_id"], queries["query_text"]))

sel, seen_q = [], {}
for name, g in bands.items():
    g = g.sample(frac=1, random_state=20260701)
    taken = 0
    for r in g.itertuples():
        if taken >= PER_BAND:
            break
        if seen_q.get(r.qid, 0) >= 3:      # limit pairs per query for diversity
            continue
        if r.a not in df.index or r.b not in df.index:
            continue
        ta, tb = str(df.loc[r.a]["title"]).strip().lower(), str(df.loc[r.b]["title"]).strip().lower()
        if ta == tb:                       # never pit a duplicate record against its twin
            continue
        seen_q[r.qid] = seen_q.get(r.qid, 0) + 1
        taken += 1
        da, db = (r.a, r.b)
        if rng.random() < 0.5:
            da, db = db, da
        sel.append({
            "pair_id": f"{r.qid}__{r.a}__{r.b}", "band": name, "query": qmap[r.qid],
            "p1_id": int(da), "p1_title": str(df.loc[da]["title"]), "p1_abstract": (df.loc[da]["abstract"] or "")[:1300],
            "p2_id": int(db), "p2_title": str(df.loc[db]["title"]), "p2_abstract": (df.loc[db]["abstract"] or "")[:1300],
        })
rng.shuffle(sel)
pd.DataFrame([{"pair_id": it["pair_id"], "band": it["band"], "p1_id": it["p1_id"], "p2_id": it["p2_id"]} for it in sel]).to_csv(
    config.REPORTS_DIR / "pairwise_review_v2_items.csv", index=False)

HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>SWRD Pairwise Review v2</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1f2733;--mut:#6b7280;--line:#e5e7eb;--accent:#1f4e79;--g:#16a34a}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;z-index:10}
.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.row h1{font-size:16px;margin:0}.stat{font-size:13px;color:var(--mut)}
.bar{height:8px;background:var(--line);border-radius:6px;overflow:hidden;margin-top:8px}.fill{height:100%;width:0;background:var(--g);transition:.2s}
.wrap{max-width:900px;margin:16px auto;padding:0 14px 130px}
.query{font-size:19px;font-weight:700;color:var(--accent);background:#eef4fb;border-left:5px solid var(--accent);padding:11px 13px;border-radius:6px;margin-bottom:6px}
.instruct{font-size:13px;color:var(--mut);margin:8px 2px 12px}
.papers{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:680px){.papers{grid-template-columns:1fr}}
.paper{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.plabel{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin-bottom:4px}
.ptitle{font-weight:600;margin:0 0 8px;font-size:15px}.pabs{font-size:13.5px;color:#374151;white-space:pre-wrap;max-height:300px;overflow:auto}
.choices{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px}
.btn{border:2px solid var(--line);background:#fff;border-radius:10px;padding:13px 8px;cursor:pointer;text-align:center;font-size:14px;font-weight:600}
.btn:hover{border-color:#cbd5e1}.btn.sel{border-color:var(--accent);background:#eef4fb}
.nav{display:flex;justify-content:space-between;margin-top:16px}.nav button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 16px;cursor:pointer}
.footer{position:fixed;bottom:0;left:0;right:0;background:var(--card);border-top:1px solid var(--line);padding:10px;display:flex;gap:10px;justify-content:center;align-items:center;flex-wrap:wrap}
.footer button{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 16px;font-size:14px;cursor:pointer}
.mapwrap{margin-top:14px}.mapwrap summary{cursor:pointer;font-size:14px;color:var(--accent);font-weight:600}
.map{display:grid;grid-template-columns:repeat(auto-fill,minmax(34px,1fr));gap:6px;margin-top:10px}
.chip{border:1.5px solid var(--line);border-radius:7px;padding:6px 0;text-align:center;font-size:12px;cursor:pointer;background:#fff;color:#374151}
.chip.done{background:#dcfce7;border-color:#16a34a;color:#14532d}
.chip.eq{background:#e0e7ff;border-color:#6366f1;color:#312e81}
.chip.cur{outline:3px solid var(--accent);outline-offset:1px;font-weight:700}
kbd{background:#eef1f4;border:1px solid #d7dbe0;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body>
<header><div class=row><h1>SWRD Pairwise Review v2</h1><span class=stat id=counter></span><span class=stat id=remain></span>
<span class=stat style="margin-left:auto">Keys: <kbd>1</kbd> left <kbd>2</kbd> right <kbd>3</kbd> equal <kbd>&larr;/&rarr;</kbd></span></div>
<div class=bar><div class=fill id=fill></div></div></header>
<div class=wrap>
<div class=query id=query></div>
<div class=instruct>Which paper better answers this search? Your progress saves automatically &mdash; you can close and reopen and it returns to where you left off.</div>
<div class=papers>
 <div class=paper><div class=plabel>Paper 1</div><p class=ptitle id=t1></p><div class=pabs id=a1></div></div>
 <div class=paper><div class=plabel>Paper 2</div><p class=ptitle id=t2></p><div class=pabs id=a2></div></div>
</div>
<div class=choices>
 <div class=btn id=b1 onclick="pick('p1')">Paper 1 better</div>
 <div class=btn id=b3 onclick="pick('equal')">About equal</div>
 <div class=btn id=b2 onclick="pick('p2')">Paper 2 better</div>
</div>
<div class=nav><button onclick="go(-1)">&larr; Prev</button><span class=stat id=saved></span><button onclick="go(1)">Next &rarr;</button></div>
<details class=mapwrap open><summary>Progress map &mdash; tap any pair to jump (green = rated, blue = "equal", outlined = current)</summary><div class=map id=map></div></details>
</div>
<div class=footer><span class=stat id=fcount></span>
<button onclick="nextUnrated()" style="background:#fff;color:#1f2733;border:1px solid #e5e7eb">Next unrated</button>
<button onclick="dl()">&#11015; Download choices (CSV)</button>
<button onclick="reset()" style="background:#fff;color:#1f2733;border:1px solid #e5e7eb">Reset</button></div>
<script>
var ITEMS=__ITEMS__;
var KEY="swrd_pairwise_v3", POSKEY="swrd_pairwise_v3_pos";
var ch={};
try{ch=JSON.parse(localStorage.getItem(KEY)||"{}")||{}}catch(e){ch={}}
function firstUnrated(){for(var k=0;k<ITEMS.length;k++){if(ch[ITEMS[k].pair_id]===undefined)return k;}return ITEMS.length-1}
var i=parseInt(localStorage.getItem(POSKEY),10);
if(!(typeof i==="number"&&i>=0&&i<ITEMS.length&&!isNaN(i))) i=firstUnrated();
function save(){try{localStorage.setItem(KEY,JSON.stringify(ch));localStorage.setItem(POSKEY,String(i));}catch(e){}}
function done(){var n=0;for(var k=0;k<ITEMS.length;k++)if(ch[ITEMS[k].pair_id]!==undefined)n++;return n}
function $(id){return document.getElementById(id)}
function jump(k){i=k;save();render()}
function buildMap(){var m=$("map");m.innerHTML="";for(var k=0;k<ITEMS.length;k++){var d=document.createElement("div");d.className="chip";d.id="chip"+k;d.textContent=(k+1);(function(kk){d.onclick=function(){jump(kk)}})(k);m.appendChild(d)}}
function updateMap(){for(var k=0;k<ITEMS.length;k++){var d=$("chip"+k);if(!d)continue;var c=ch[ITEMS[k].pair_id];d.className="chip"+(c==="equal"?" eq":(c!==undefined?" done":""))+(k===i?" cur":"")}}
function render(){var it=ITEMS[i];var c=ch[it.pair_id];
 $("query").textContent=it.query;
 $("t1").textContent=it.p1_title;$("a1").textContent=it.p1_abstract||"(no abstract)";
 $("t2").textContent=it.p2_title;$("a2").textContent=it.p2_abstract||"(no abstract)";
 $("b1").classList.toggle("sel",c===String(it.p1_id));
 $("b2").classList.toggle("sel",c===String(it.p2_id));
 $("b3").classList.toggle("sel",c==="equal");
 $("counter").textContent="Pair "+(i+1)+" / "+ITEMS.length;
 $("remain").textContent="- "+done()+" done, "+(ITEMS.length-done())+" left";
 $("fill").style.width=(100*done()/ITEMS.length)+"%";
 $("fcount").textContent=done()+" of "+ITEMS.length+" rated";
 updateMap();save();window.scrollTo(0,0)}
function pick(v){var it=ITEMS[i];
 ch[it.pair_id]=(v==="equal")?"equal":(v==="p1"?String(it.p1_id):String(it.p2_id));
 if(i<ITEMS.length-1)i++;
 save();render()}
function go(d){i=Math.min(ITEMS.length-1,Math.max(0,i+d));save();render()}
function nextUnrated(){for(var k=1;k<=ITEMS.length;k++){var j=(i+k)%ITEMS.length;if(ch[ITEMS[j].pair_id]===undefined){i=j;save();render();return}}alert("All "+ITEMS.length+" pairs rated - tap Download choices.")}
function dl(){var rows=[["pair_id","chosen_paper_id"]];for(var k=0;k<ITEMS.length;k++){var it=ITEMS[k];if(ch[it.pair_id]!==undefined)rows.push([it.pair_id,ch[it.pair_id]])}
 var csv=rows.map(function(r){return r.join(",")}).join("\n");
 var b=new Blob([csv],{type:"text/csv"});var a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="pairwise_review_v2_completed.csv";document.body.appendChild(a);a.click();a.remove()}
function reset(){if(confirm("Clear all your choices?")){ch={};i=0;try{localStorage.removeItem(KEY);localStorage.removeItem(POSKEY)}catch(e){}render()}}
buildMap();
document.addEventListener("keydown",function(e){if(e.key==="1")pick("p1");else if(e.key==="2")pick("p2");else if(e.key==="3")pick("equal");else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1)});
render();
</script></body></html>"""

out = config.REPORTS_DIR / "pairwise_review_v2.html"
out.write_text(HTML.replace("__ITEMS__", json.dumps(sel, ensure_ascii=False)), encoding="utf-8")
from collections import Counter
print(f"wrote {out} with {len(sel)} pairs | bands: {dict(Counter(it['band'] for it in sel))}")
