
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

seed = 67
rng = np.random.default_rng(seed)
MAX_BOOKS = 1000 # MAX_BOOKS < 10001
MAX_USERS = 5000 # MAX_USERS < 53425

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

print("\nTrain:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# 희소행렬 함수
# param TF=False-> 이미 본 책 마킹, True-> 선호 책 마킹
def make_matrix(data, TF=False):
    if TF:
        data = data[data["rating"] >= 4]
    rows = data["user_id"].map(user_index).to_numpy()
    cols = data["book_id"].map(book_index).to_numpy()
    values = np.ones(len(data), dtype=np.float32)
    M = csr_matrix((values, (rows, cols)),shape=(n_users, n_books))
    return M

#Apriori 함수
def do_apriori(M, min_support=0.02, min_confidence=0.3):
    baskets = pd.DataFrame.sparse.from_spmatrix(M, columns=book_ids.astype(str)).astype(pd.SparseDtype(bool, False))
    frequent = apriori(baskets, min_support=min_support, use_colnames=True, low_memory=True)

    if frequent.empty or not frequent["itemsets"].apply(len).ge(2).any():
        return frequent, pd.DataFrame()
    
    rules = association_rules(frequent, num_itemsets=len(baskets), metric="confidence", min_threshold=min_confidence)
    rules["interest"] = (rules["confidence"] - rules["consequent support"]).abs()
    rules = rules[["antecedents", "consequents", "support", "confidence", "interest"]]
    rules = rules.sort_values("interest", ascending=False).reset_index(drop=True)
    return frequent, rules

#user-based & item-based CF
#mode 0 = user, 1 = item
def do_CF(M, mode=0):
    if mode == 1 :
        M = M.T
    similarity = cosine_similarity(M)
    np.fill_diagonal(similarity, 0)
    return similarity

#k-means clustering
#mode 0 = user, 1 = item
def do_Clustering(M, k, mode=0):
    if mode == 1 :
        M = M.T
    book_vectors = normalize(M.tocsr(), norm="l2", axis=1)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(book_vectors)
    return labels

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

#k-means 기반 item-based CF
def do_Cluster_CF(similarity, labels):
    same_cluster = labels[:, None] == labels[None, :]
    cluster_similarity = similarity * same_cluster
    return cluster_similarity

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


M_liked = make_matrix(train, TF=True)
M_seen = make_matrix(train, TF=False)

frequent, rules = do_apriori(M_liked)
print("\n빈발 항목집합 개수:", len(frequent))
print("연관 규칙 개수:", len(rules))
print(rules.head(5))

similarity_user = do_CF(M_liked, mode=0)
similarity_item = do_CF(M_liked, mode=1)
labels_user = do_Clustering(M_liked, k=10, mode=0)
labels_item = do_Clustering(M_liked, k=10, mode=1)
print("\n클러스터별 인원:")
print(pd.Series(labels_user).value_counts().sort_index())
print("\n클러스터별 도서 수:")
print(pd.Series(labels_item).value_counts().sort_index())

cluster_similarity_user = do_Cluster_CF(similarity_user, labels_user)
cluster_similarity_item = do_Cluster_CF(similarity_item, labels_item)
tag_scores = find_genre(labels_item)
user = int(user_ids[10])
titles = books.set_index("book_id")["title"].to_dict()

recom_user = recommend_CF(user, M_liked, M_seen, similarity_user, top_k=10, mode=0)
recom_item = recommend_CF(user, M_liked, M_seen, similarity_item, top_k=10, mode=1)
recom_cluster_user = recommend_CF(user, M_liked, M_seen, cluster_similarity_user, top_k=10, mode=0)
recom_cluster_item = recommend_CF( user, M_liked, M_seen,cluster_similarity_item, top_k=10, mode=1)

results = {
    "User-based CF": recom_user,
    "Item-based CF": recom_item,
    "K-means + User-based CF": recom_cluster_user,
    "K-means + Item-based CF": recom_cluster_item
}

for i, recommendations in results.items():
    print(f"\n{i} 추천 도서:")
    for book_id in recommendations:
        print(book_id, titles.get(book_id))

print("\n특징 태그:")
for cluster in tag_scores.index:
    print(f"\nCluster {cluster}")
    print(tag_scores.loc[cluster].nlargest(3))