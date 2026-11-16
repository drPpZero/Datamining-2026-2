# 도서 추천 시스템 구현
### 팀원: 한지우, 신서윤, 손민서

## 프로젝트 소개
Goodbooks-10k 데이터셋을 활용해 적절한 도서를 추천하는 추천 시스템을 구현하였습니다.

Collaborative Filtering, Apriori, K-mean Clustering 알고리즘을 적용하여 추천 시스템을 구현하였습니다.

## Repository 구조
```
Datamining-2026-2/
├── .gitignore
├── README.md
├── main.py
├── data/
    ├── ratings.csv
    ├── books.csv
    ├── book_tags.csv
    └── tags.csv
└── results/
```
## Dataset

Goodbooks-10k
출처: https://github.com/zygmuntz/goodbooks-10k
라이선스: CC BY-SA 4.0
데이터셋을 다운로드 후, 이 프로젝트의 data 폴더에 넣어주세요.

## 실행 방법
데이터셋을 다운로드하여, data 폴더 내에 넣습니다.

라이브러리를 설치하기 위해, 아래 명령어를 터미널에 입력합니다.
```
pip install numpy pandas scipy scikit-learn mlxtend matplotlib umap-learn
```
이후 프로젝트 루트 디렉토리에서,
```
python main.py
```
를 입력해 실행할 수 있습니다.

시드값과 책, 사용자 수를 변경하고 싶다면 
```
python main.py --seed 1111 --books 2222 --users 3333
```
과 같이 입력 후 실행할 수 있습니다.

기본 값은 seed= , books=2000, users=10000 입니다.