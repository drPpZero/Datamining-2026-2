import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap
import argparse
from pathlib import Path

#파라미터 및 시드 설정
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9875)
parser.add_argument("--books", type=int, default=2000)
parser.add_argument("--users", type=int, default=10000)
args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed)
MAX_BOOKS = args.books # MAX_BOOKS < 10001
MAX_USERS = args.users # MAX_USERS < 53425

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)

"""
데이터 전처리
Goodbooks-10k 데이터셋을 가져와 설정한 하이퍼파라미터(MAX_BOOKS, MAX_USERS)만큼 사용
Train, Validation, Test set으로 8:1:1 비율로 분할
"""
ratings = pd.read_csv("data/ratings.csv")
books = pd.read_csv("data/books.csv")
book_tags = pd.read_csv("data/book_tags.csv")
tags = pd.read_csv("data/tags.csv")

ratings = ratings.drop_duplicates(["user_id", "book_id"]).reset_index(drop=True)

selected_books = (ratings["book_id"].value_counts().head(MAX_BOOKS).index)
ratings = ratings[ratings["book_id"].isin(selected_books)].copy()

user_counts = ratings["user_id"].value_counts()
selected_users = (user_counts[user_counts >= 10].head(MAX_USERS).index)
ratings = ratings[ratings["user_id"].isin(selected_users)].reset_index(drop=True)

user_ids = np.sort(ratings["user_id"].unique())
book_ids = np.sort(ratings["book_id"].unique())
n_users = len(user_ids)
n_books = len(book_ids)
user_index = {user_id: i for i, user_id in enumerate(user_ids)}
book_index = {book_id: i for i, book_id in enumerate(book_ids)}

set_val = []
set_test = []

for user_id, group in ratings.groupby("user_id"):
    indices = group.index.to_numpy().copy()
    rng.shuffle(indices)
    n_val = max(1, round(len(indices) * 0.1))
    n_test = max(1, round(len(indices) * 0.1))
    set_val.extend(indices[:n_val])
    set_test.extend(indices[n_val:n_val + n_test])

train = ratings.drop(index=set_val + set_test).reset_index(drop=True)
val = ratings.loc[set_val].reset_index(drop=True)
test = ratings.loc[set_test].reset_index(drop=True)

titles = books.set_index("book_id")["title"].to_dict()

print("\nTrain:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

'''
알고리즘 구현 및 태그 분석, 시각화, 평가 함수
알고리즘은 Apriori, User/Item-based CF, k-mean Clustering
Cluster 시각화는 UMAP 사용
비교용 Baseline: Random, Popularity
평가 지표는 Recall@K. NDCG@K
'''
# 희소행렬 함수
# param TF=False-> 이미 본 책 마킹, True-> 선호 책 마킹
def make_matrix(data, TF=False):
    if TF:
        data = data[data["rating"] >= 4]
    rows = data["user_id"].map(user_index).to_numpy()
    cols = data["book_id"].map(book_index).to_numpy()
    values = np.ones(len(data), dtype=np.float32)
    M = csr_matrix((values, (rows, cols)), shape=(n_users, n_books))
    return M

#Apriori 함수
def do_apriori(M, min_support=0.03, min_confidence=0.3):
    baskets = pd.DataFrame.sparse.from_spmatrix(M, columns=book_ids.astype(str)).astype(pd.SparseDtype(bool, False))
    frequent = apriori(baskets, min_support=min_support, use_colnames=True, low_memory=True, max_len=3)

    if frequent.empty or not frequent["itemsets"].apply(len).ge(2).any():
        return frequent, pd.DataFrame()
    
    rules = association_rules(frequent, num_itemsets=len(baskets), metric="confidence", min_threshold=min_confidence)
    rules["interest"] = (rules["confidence"] - rules["consequent support"]).abs()
    rules = rules[["antecedents", "consequents", "support", "confidence", "interest"]]
    rules = rules.sort_values("interest", ascending=False).reset_index(drop=True)
    return frequent, rules

#연관규칙 재사용을 위한 저장 함수
def save_rules(rules):
    rule_index = {}
    for ant, con, confidence in rules[["antecedents", "consequents", "confidence"]].itertuples(index=False, name=None):
        book = next(iter(ant))
        if book not in rule_index:
            rule_index[book] = []
        rule_index[book].append((ant, con, confidence))
    return rule_index

#Apriori 기반 추천 시스템
def recom_Apriori(user_id, M, M_seen, rule_index, top_k=10):
    u = user_index[user_id]
    liked = set(book_ids[M[u].indices].astype(str))
    seen = set(book_ids[M_seen[u].indices].astype(str))
    scores = {}
    for book in liked:
        for ant, con, confidence in rule_index.get(book, []):
            if ant.issubset(liked):
                for target in con:
                    if target not in seen:
                        scores[target] = max(scores.get(target, 0), confidence)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [int(book) for book in ranked[:top_k]]

#user-based & item-based CF
#mode 0 = user, 1 = item
def do_CF(M, mode=0):
    if mode == 1:
        M = M.T
    similarity = cosine_similarity(M)
    np.fill_diagonal(similarity, 0)
    return similarity

#k-means clustering
#mode 0 = user, 1 = item
def do_Clustering(M, k, mode=0):
    if mode == 1:
        M = M.T
    book_vectors = normalize(M.tocsr(), norm="l2", axis=1)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(book_vectors)
    return labels

#클러스터 시각화(UMAP) 함수
#mode 0 = user, 1 = item
def plot_UMAP(M, labels, mode=0):
    if mode == 1:
        M = M.T
    vectors = normalize(M.tocsr(), norm="l2", axis=1)
    model = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=seed)
    points = model.fit_transform(vectors)
    fig = plt.figure(figsize=(9, 7))
    n_clusters = len(np.unique(labels))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    cmap = plt.matplotlib.colors.ListedColormap(colors[:n_clusters])
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, vmin=-0.5, vmax=n_clusters-0.5, s=12, alpha=0.8)
    plt.colorbar(scatter, ticks=np.arange(n_clusters), label="Cluster")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("User Clustering" if mode == 0 else "Book Clustering")
    plt.tight_layout()
    plt.tight_layout()
    return fig

# Cosine similarity 기반 도서 추천
# mode 0 = user, 1 = item
def recommend_CF(user_id, M, M_seen, similarity, top_k=5, mode=0):
    u = user_index[user_id]
    if mode == 0:
        scores = np.asarray(similarity[u] @ M).ravel()
    else:
        scores = np.asarray(M[u] @ similarity).ravel()
    seen = M_seen[u].indices
    scores[seen] = -np.inf
    ranked = np.argsort(-scores)
    ranked = [i for i in ranked if scores[i] > 0][:top_k]
    return [int(book_ids[i]) for i in ranked]

#Baseline 1: 랜덤 추천
def recommend_random(user_id, M_seen, top_k=10):
    u = user_index[user_id]
    candidates = np.setdiff1d(np.arange(n_books), M_seen[u].indices)
    rng_user = np.random.default_rng(seed + int(user_id))
    selected = rng_user.choice(candidates, size=min(top_k, len(candidates)), replace=False)
    return [int(book_ids[i]) for i in selected]

#Baseline 2: 인기 순 추천
def recommend_popular(user_id, M_seen, popularity_scores, top_k=10):
    u = user_index[user_id]
    scores = popularity_scores.copy()
    scores[M_seen[u].indices] = -np.inf
    ranked = np.argsort(-scores)
    ranked = [i for i in ranked if scores[i] > 0][:top_k]
    return [int(book_ids[i]) for i in ranked]

#k-means 기반 item-based CF
def do_Cluster_CF(similarity, labels):
    same_cluster = labels[:, None] == labels[None, :]
    cluster_similarity = similarity * same_cluster
    return cluster_similarity

#클러스터 별 빈발 태그 탐색 함수
def find_genre(labels):
    book_clusters = pd.DataFrame({"book_id": book_ids, "cluster": labels})
    tag_data = (book_clusters
        .merge(books[["book_id", "goodreads_book_id"]], on="book_id")
        .merge(book_tags, on="goodreads_book_id")
        .merge(tags, on="tag_id"))
    tag_data = tag_data[["book_id", "cluster", "tag_name"]].drop_duplicates()
    cluster_sizes = book_clusters["cluster"].value_counts().sort_index()
    tag_counts = pd.crosstab(tag_data["cluster"], tag_data["tag_name"])
    cluster_ratio = tag_counts.div(cluster_sizes, axis=0)
    overall_ratio = (tag_data.groupby("tag_name")["book_id"].nunique() / n_books)
    tag_scores = cluster_ratio.div(overall_ratio, axis=1)
    for cluster in tag_scores.index:
        tag_scores.loc[cluster, tag_counts.loc[cluster] < 3] = 0
    return tag_scores

#태그 시각화용 함수
def make_tag_table(tag_scores, labels, top_n=3):
    rows = []

    for cluster in tag_scores.index:
        top_tags = tag_scores.loc[cluster]
        top_tags = top_tags[top_tags > 0].sort_values(ascending=False).head(top_n)

        row = {"Cluster": cluster, "Count": int(np.sum(labels == cluster))}
        for i, (tag, score) in enumerate(top_tags.items(), start=1):
            row[f"Tag{i}"] = tag
            row[f"Score{i}"] = round(score, 3)
        rows.append(row)
    tag_table = pd.DataFrame(rows)

    return tag_table

    
#평가 함수
#Recall@K, NDCG@K 사용
def evaluate(func, data, top_k=10):
    likes = (data[data["rating"] >= 4].groupby("user_id")["book_id"].agg(set))
    recalls = []
    ndcgs = []

    for user_id, relevant in likes.items():
        recom = func(user_id)[:top_k]
        # Recall@K
        hits = len(set(recom) & relevant)
        recall = hits / len(relevant)
        # NDCG@K
        dcg = 0
        for rank, book_id in enumerate(recom, start=1):
            if book_id in relevant:
                dcg += 1 / np.log2(rank + 1)
        idcg = 0
        for rank in range(1, min(len(relevant), top_k) + 1):
            idcg += 1 / np.log2(rank + 1)
        ndcg = dcg / idcg
        recalls.append(recall)
        ndcgs.append(ndcg)
    return {
        "Recall@10": np.mean(recalls),
        "NDCG@10": np.mean(ndcgs),
        "Users": len(likes)
    }

'''
실행 코드
흐름은 아래와 같음

validation set 이용하여 k-mean 최적 k값 도출
->이후 train+validation set으로 재계산
->생성된 클러스터 별 빈발 태그 분석
->Apriori 연관규칙 생성 및 저장
->test set 이용하여 Baseline + 추천 시스템 5가지의 책 추천 평가
-> 결과 및 UMAP 출력
-> 종료
'''
M_liked = make_matrix(train, TF=True)
M_seen = make_matrix(train, TF=False)
user = int(user_ids[10])

similarity_user = do_CF(M_liked, mode=0)
similarity_item = do_CF(M_liked, mode=1)

k_values = [10, 20, 30, 40, 50]
k_results = []

for k in k_values:
    labels_user = do_Clustering(M_liked, k=k, mode=0)
    labels_item = do_Clustering(M_liked, k=k, mode=1)
    cluster_similarity_user = do_Cluster_CF(similarity_user, labels_user)
    cluster_similarity_item = do_Cluster_CF(similarity_item, labels_item)

    result_user = evaluate(lambda u: recommend_CF(u, M_liked, M_seen, cluster_similarity_user, top_k=10, mode=0), val, top_k=10)
    result_item = evaluate(lambda u: recommend_CF(u, M_liked, M_seen, cluster_similarity_item, top_k=10, mode=1), val, top_k=10)

    k_results.append([k, result_user["NDCG@10"], result_item["NDCG@10"]])
    print(f"k={k} 완료", flush=True)

k_results_df = pd.DataFrame(k_results, columns=["k", "User_NDCG@10", "Item_NDCG@10"])
best_k_user = int(k_results_df.loc[k_results_df["User_NDCG@10"].idxmax(), "k"])
best_k_item = int(k_results_df.loc[k_results_df["Item_NDCG@10"].idxmax(), "k"])

print("\n최적 k 찾기:")
print(k_results_df.round(4))
print("최적 User k:", best_k_user)
print("최적 Item k:", best_k_item)

train_final = pd.concat([train, val], ignore_index=True)
M_liked = make_matrix(train_final, TF=True)
M_seen = make_matrix(train_final, TF=False)
similarity_user = do_CF(M_liked, mode=0)
similarity_item = do_CF(M_liked, mode=1)
labels_user = do_Clustering(M_liked, k=best_k_user, mode=0)
labels_item = do_Clustering(M_liked, k=best_k_item, mode=1)
cluster_similarity_user = do_Cluster_CF(similarity_user, labels_user)
cluster_similarity_item = do_Cluster_CF(similarity_item, labels_item)

print("태그 비교 중...")
tag_scores = find_genre(labels_item)
tag_table = make_tag_table(tag_scores, labels_item, top_n=3)
print("연관 규칙 생성 중...")
frequent, rules = do_apriori(M_liked)
rule_index = save_rules(rules) if not rules.empty else {}

popularity_scores = np.asarray(M_liked.sum(axis=0)).ravel()

funcs = {
    "Random Baseline": lambda u: recommend_random(u, M_seen, top_k=10),
    "Popularity Baseline": lambda u: recommend_popular(u, M_seen, popularity_scores, top_k=10),
    "User-based CF": lambda u: recommend_CF(u, M_liked, M_seen, similarity_user, top_k=10, mode=0),
    "Item-based CF": lambda u: recommend_CF(u, M_liked, M_seen, similarity_item, top_k=10, mode=1),
    "K-means + User CF": lambda u: recommend_CF(u, M_liked, M_seen, cluster_similarity_user, top_k=10, mode=0),
    "K-means + Item CF": lambda u: recommend_CF(u, M_liked, M_seen, cluster_similarity_item, top_k=10, mode=1),
    "Apriori": lambda u: recom_Apriori(u, M_liked, M_seen, rule_index, top_k=10)
}

results = []
for name, func in funcs.items():
    print("평가 중: " + name)
    result = evaluate(func, test, top_k=10)
    result["Method"] = name
    results.append(result)

results_df = pd.DataFrame(results).set_index("Method")
print("\nTest 평가 결과:")
print(results_df.round(4))
fig_user = plot_UMAP(M_liked, labels_user, mode=0)
fig_item = plot_UMAP(M_liked, labels_item, mode=1)

print("\n결과 저장 중...")
tag_table.to_csv(results_dir / "cluster_tags.csv", index=False, encoding="utf-8-sig")
k_results_df.to_csv(results_dir / "k_validation.csv", index=False, encoding="utf-8-sig")
results_df.to_csv(results_dir / "test_results.csv", encoding="utf-8-sig")
fig_user.savefig(results_dir / "umap_users.png", dpi=200, bbox_inches="tight")
fig_item.savefig(results_dir / "umap_books.png", dpi=200, bbox_inches="tight")
print("\n결과 저장 완료")
