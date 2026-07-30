"""Ablation 数据分析：Reflect ON vs OFF"""
import json, os

def load(base_dir, ids):
    raw = os.path.join(base_dir, "raw")
    r = {}
    for f in os.listdir(raw):
        cid = f.replace(".json", "")
        if cid in ids:
            d = json.load(open(os.path.join(raw, f), encoding="utf-8"))
            s = d.get("score", {})
            r[cid] = {
                "total": s.get("total_score", 0),
                "comp": s.get("completeness", {}).get("score", 0),
                "cite": s.get("citation_quality", {}).get("score", 0),
                "rel": s.get("relevance", {}).get("score", 0),
                "evidence": len(d.get("evidence", {}).get("evidence_pool", [])),
                "bocha": d.get("bocha_calls", 0),
                "llm": d.get("llm_calls", 0),
                "elapsed": d.get("elapsed", 0),
            }
    return r

ids = {"MR-01", "MR-05", "TA-02", "TA-04", "KQ-01", "KQ-06"}
off = load("output/eval_ablation_reflect_off", ids)
on = load("output/eval_ablation_reflect_on", ids)

order = ["MR-01", "MR-05", "TA-02", "TA-04", "KQ-01", "KQ-06"]
all_deltas = {"total": [], "evidence": [], "bocha": [], "llm": [], "elapsed": []}

print("=" * 110)
print("ABLATION STUDY: Reflect ON (max_iterations=2) vs OFF (max_iterations=0)")
print("=" * 110)
print()
h = f"{'Case':<8} {'Mode':<8} {'Score':<8} {'Comp':<8} {'Cite':<8} {'Rel':<8} {'Evid':<6} {'Bocha':<6} {'LLM':<6} {'Time':<8} {'Eff':<8}"
print(h)
print("-" * len(h))

for cid in order:
    ro = on.get(cid, {})
    rf = off.get(cid, {})
    eff_on = ro.get("evidence", 0) / max(ro.get("bocha", 1), 1)
    eff_off = rf.get("evidence", 0) / max(rf.get("bocha", 1), 1)
    print(f"{cid:<8} {'ON':<8} {ro.get('total','?'):<8} {ro.get('comp','?'):<8} {ro.get('cite','?'):<8} {ro.get('rel','?'):<8} {ro.get('evidence','?'):<6} {ro.get('bocha','?'):<6} {ro.get('llm','?'):<6} {ro.get('elapsed','?'):<8}s {eff_on:<.1f}")
    print(f"{'':<8} {'OFF':<8} {rf.get('total','?'):<8} {rf.get('comp','?'):<8} {rf.get('cite','?'):<8} {rf.get('rel','?'):<8} {rf.get('evidence','?'):<6} {rf.get('bocha','?'):<6} {rf.get('llm','?'):<6} {rf.get('elapsed','?'):<8}s {eff_off:<.1f}")
    dt = ro.get("total", 0) - rf.get("total", 0)
    print(f"{'':<8} {'Δ':<8} {dt:<+8.1f} {ro.get('comp',0)-rf.get('comp',0):<+8.1f} {ro.get('cite',0)-rf.get('cite',0):<+8.1f} {ro.get('rel',0)-rf.get('rel',0):<+8.1f} {ro.get('evidence',0)-rf.get('evidence',0):<+6.0f} {ro.get('bocha',0)-rf.get('bocha',0):<+6.1f} {ro.get('llm',0)-rf.get('llm',0):<+6.1f} {ro.get('elapsed',0)-rf.get('elapsed',0):<+8.0f}s {eff_on-eff_off:<+.1f}")
    for k in all_deltas:
        if ro.get(k) is not None and rf.get(k) is not None:
            all_deltas[k].append(ro[k] - rf[k])
    print()

print("=" * 110)
print("SUMMARY")
print("-" * 60)

all_on = [on[c] for c in order]
all_off = [off[c] for c in order]

def avg(lst, key):
    return sum(item.get(key, 0) for item in lst) / len(lst)

labels = {"total": "Score", "evidence": "Evidence", "bocha": "Bocha", "llm": "LLM", "elapsed": "Time(s)"}
print(f"{'Metric':<12} {'ON':<10} {'OFF':<10} {'Δ':<10} {'Δ%':<10}")
print("-" * 52)
for k, lbl in labels.items():
    vo = avg(all_on, k)
    vf = avg(all_off, k)
    d = vo - vf
    pct = (d / vf * 100) if vf != 0 else 0
    print(f"{lbl:<12} {vo:<10.2f} {vf:<10.2f} {d:<+10.2f} {pct:<+10.1f}%")

# ROI
ds = avg(all_on, "total") - avg(all_off, "total")
dllm = avg(all_on, "llm") - avg(all_off, "llm")
db = avg(all_on, "bocha") - avg(all_off, "bocha")
print()
print("ROI Analysis:")
print(f"  Score gain per extra LLM call:   {ds/dllm:.2f}" if dllm else "  N/A")
print(f"  Score gain per extra Bocha call: {ds/db:.2f}" if db else "  N/A")
