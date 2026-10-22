import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

df = pd.read_csv("customers.csv")          # <- change to your own CSV
print("Shape:", df.shape); print(df.isna().sum().to_string()); print(df.describe().round(2).to_string())

features = ["Age", "Annual_Income_k", "Spending_Score"]
X = StandardScaler().fit_transform(df[features])

# Choose k: elbow (inertia) + silhouette
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}  inertia={km.inertia_:.1f}  silhouette={sil[-1]:.3f}")
best_k = list(ks)[int(np.argmax(sil))]
print("Best k by silhouette:", best_k)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sil, "o-", c="green"); ax[1].set(title="Silhouette score", xlabel="k")
plt.tight_layout(); plt.savefig("fig_k.png", dpi=150); plt.close()

# Final K-Means model
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["Cluster"] = km.labels_

# Compare with other algorithms
hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
print("\nModel comparison")
print(f"KMeans        silhouette={silhouette_score(X, km.labels_):.3f}  DB={davies_bouldin_score(X, km.labels_):.3f}")
print(f"Hierarchical  silhouette={silhouette_score(X, hc):.3f}  DB={davies_bouldin_score(X, hc):.3f}")
m = db != -1
print(f"DBSCAN        clusters={len(set(db))-(1 if -1 in db else 0)}  noise={(~m).sum()}  silhouette={silhouette_score(X[m], db[m]):.3f}")

# Segment profile
profile = df.groupby("Cluster").agg(Customers=("CustomerID","count"), Avg_Age=("Age","mean"),
          Avg_Income_k=("Annual_Income_k","mean"), Avg_Spend=("Spending_Score","mean")).round(1)
profile["Share_%"] = (profile.Customers/len(df)*100).round(1)
print("\nSegment profile"); print(profile.to_string())
print("\nGender mix (%)"); print((pd.crosstab(df.Cluster, df.Gender, normalize="index")*100).round(1).to_string())

# Plots
pcs = PCA(n_components=2).fit(X); P = pcs.transform(X)
print("\nPCA explained variance:", pcs.explained_variance_ratio_.round(3))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sc = ax[0].scatter(df.Annual_Income_k, df.Spending_Score, c=df.Cluster, cmap="tab10", s=18)
ax[0].set(title="Income vs Spending Score", xlabel="Annual income (k$)", ylabel="Spending score")
ax[1].scatter(P[:,0], P[:,1], c=df.Cluster, cmap="tab10", s=18)
ax[1].set(title="PCA projection of clusters", xlabel="PC1", ylabel="PC2")
for c in range(best_k): ax[0].annotate(str(c), profile.loc[c, ["Avg_Income_k","Avg_Spend"]].values, fontsize=13, weight="bold")
plt.tight_layout(); plt.savefig("fig_clusters.png", dpi=150); plt.close()
df.to_csv("customers_segmented.csv", index=False)
