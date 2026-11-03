"""Print the week-2 clustering quality numbers as one table (reads results only, writes nothing)."""
import json

PATH = "outputs/results/week_02_dev_and_clusters/cluster_metrics.json"
m = json.load(open(PATH, encoding="utf-8"))

print("pairs clustered:", m["n_pairs"])
print(f"{'model':6} {'k':>3} {'silhouette':>10} {'contig_mendeley':>16} {'contig_chauhan':>15} {'ARI':>6}")
for model, by_k in m["silhouette"].items():
    for k, sil in by_k.items():
        c = m["contiguity"][model][k]
        print(f"{model:6} {k:>3} {sil:10.4f} {str(c['mendeley']['ratio']):>16} "
              f"{str(c['chauhan']['ratio']):>15} {m['agreement_ari'][k]:6}")