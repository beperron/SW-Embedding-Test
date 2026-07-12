"""Compare the same-family committee (Qwen3.6 + Gemma) vs the neutral frontier committee (GLM-5.2 +
Kimi-K2.6) pairwise leaderboards. Tests the same-family-bias hypothesis: do the Qwen-embedding and
EmbeddingGemma candidates rank higher under the same-family judges than under neutral judges?"""
import pandas as pd
from scipy.stats import spearmanr
from swrd_eval import config
sf = pd.read_parquet(config.RUNS_DIR/"metrics_pairwise.parquet")[["system","nDCG@10"]].rename(columns={"nDCG@10":"samefam"})
fr = pd.read_parquet(config.RUNS_DIR/"metrics_pairwise_frontier.parquet")[["system","nDCG@10"]].rename(columns={"nDCG@10":"frontier"})
m = sf.merge(fr,on="system")
m["rank_sf"]=m["samefam"].rank(ascending=False); m["rank_fr"]=m["frontier"].rank(ascending=False)
m["rank_shift"]=m["rank_sf"]-m["rank_fr"]   # positive = ranked BETTER (lower number) by frontier
rho,_=spearmanr(m["samefam"],m["frontier"])
print(f"leaderboard rank correlation same-family vs frontier: Spearman rho = {rho:.4f}\n")
dn=m[m.system.str.startswith("dense.")].copy(); dn["mdl"]=dn.system.str.replace("dense.","")
dn=dn.sort_values("samefam",ascending=False)
print(f"{'model':18}{'rank(SF)':>9}{'rank(FR)':>9}{'shift':>7}  {'nDCG SF':>8}{'nDCG FR':>8}")
for _,r in dn.iterrows():
    fam = "  <- Qwen/Gemma family" if any(k in r.mdl for k in ["qwen3-emb","embeddinggemma"]) else ""
    print(f"{r.mdl:18}{r.rank_sf:9.0f}{r.rank_fr:9.0f}{r.rank_sf-r.rank_fr:+7.0f}  {r.samefam:8.4f}{r.frontier:8.4f}{fam}")
print("\n(rank shift > 0 means the neutral frontier judges ranked it BETTER than the same-family judges;")
print(" < 0 means the same-family judges ranked it better -> possible same-family favoritism.)")
