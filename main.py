
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules

seed = 67
rng = np.random.default_rng(seed)
MAX_BOOKS = 5000 # MAX_BOOKS < 10001
MAX_USERS = 30000 # MAX_USERS < 53425

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
# param TF는 4 이상 평점을 1, 3 이하 평점을 0으로 치환
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
    frequent = apriori(baskets, min_support=min_support, use_colnames=True, max_len=2, low_memory=True)

    if frequent.empty or not frequent["itemsets"].apply(len).ge(2).any():
        return frequent, pd.DataFrame()
    
    rules = association_rules(frequent, num_itemsets=len(baskets), metric="confidence", min_threshold=min_confidence)
    rules["interest"] = (rules["confidence"] - rules["consequent support"]).abs()
    rules = rules[["antecedents", "consequents", "support", "confidence", "interest"]]
    rules = rules.sort_values("interest", ascending=False).reset_index(drop=True)
    return frequent, rules

matrix = make_matrix(train, TF=True)
frequent, rules = do_apriori(matrix)

print("빈발 항목집합 개수:", len(frequent))
print("연관 규칙 개수:", len(rules))
print(rules.head(10))