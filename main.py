
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity

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

#Item-based CF 
#Cosine similarity 계산 함수
def do_CF(M):
    similarity = cosine_similarity(M.T)
    np.fill_diagonal(similarity, 0)
    return similarity

#Cosine similarity 기반 도서 추천
def recommend_CF(user_id, M, M_seen, similarity, top_k=5):
    u = user_index[user_id]
    scores = np.asarray(M[u] @ similarity).ravel()
    seen = M_seen[u].indices
    scores[seen] = -np.inf
    ranked = np.argsort(-scores)
    ranked = [i for i in ranked if scores[i] > 0][:top_k]
    return [int(book_ids[i]) for i in ranked]

def do_Clustering():
    return

matrix = make_matrix(train, TF=True)
frequent, rules = do_apriori(matrix)

#print("빈발 항목집합 개수:", len(frequent))
#print("연관 규칙 개수:", len(rules))
#print(rules.head(10))


M_liked = make_matrix(train, TF=True)
M_seen = make_matrix(train, TF=False)
similarity = do_CF(M_liked)
user = user_ids[0]
recom = recommend_CF(user, M_liked, M_seen, similarity)
titles = books.set_index("book_id")["title"].to_dict()

print("\n추천 도서:")
for book_id in recom:
    print(book_id, titles.get(book_id))