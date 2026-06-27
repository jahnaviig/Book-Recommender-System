"""
train_model.py
----------------
Builds two recommendation systems from the Book-Crossing dataset:

1. Popularity-based  -> top 50 books by average rating (min 250 ratings)
2. Collaborative filtering -> "books similar to this one" using cosine
   similarity on a user-book ratings pivot table

IMPORTANT: this script must be run once (from the project root, with the
`datasets/` folder containing Books.csv, Users.csv, Ratings.csv) before
starting web.py, because web.py loads the .pkl files this script produces.

Run:
    python train_model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"))
users = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"))

print("Books:", books.shape)
print("Users:", users.shape)
print("Ratings:", ratings.shape)

# Quick sanity checks (kept from the original notebook-style exploration)
print("\nMissing values:")
print("books  ->", books.isnull().sum().sum())
print("users  ->", users.isnull().sum().sum())
print("ratings->", ratings.isnull().sum().sum())

print("\nDuplicate rows:")
print("books  ->", books.duplicated().sum())
print("users  ->", users.duplicated().sum())
print("ratings->", ratings.duplicated().sum())

# ---------------------------------------------------------------------------
# 2. Popularity-based recommender
# ---------------------------------------------------------------------------
ratings_with_name = ratings.merge(books, on="ISBN")

num_rating_df = (
    ratings_with_name.groupby("Book-Title")
    .count()["Book-Rating"]
    .reset_index()
    .rename(columns={"Book-Rating": "num_ratings"})
)

ratings_with_name["Book-Rating"] = pd.to_numeric(
    ratings_with_name["Book-Rating"], errors="coerce"
)
ratings_with_name.dropna(subset=["Book-Rating"], inplace=True)
ratings_with_name["Book-Rating"] = ratings_with_name["Book-Rating"].astype(float)

avg_rating_df = (
    ratings_with_name.groupby("Book-Title")["Book-Rating"]
    .mean()
    .reset_index()
    .rename(columns={"Book-Rating": "avg_rating"})
)

popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
popular_df = popular_df[popular_df["num_ratings"] >= 250].sort_values(
    "avg_rating", ascending=False
)
popular_df = (
    popular_df.merge(books, on="Book-Title")
    .drop_duplicates("Book-Title")[
        ["Book-Title", "Book-Author", "Image-URL-M", "num_ratings", "avg_rating"]
    ]
)
popular_df.reset_index(drop=True, inplace=True)

print("\nTop 50 popular books ready:", popular_df.shape)

# ---------------------------------------------------------------------------
# 3. Collaborative-filtering recommender
# ---------------------------------------------------------------------------
# Keep only users who have rated more than 200 books ("engaged" readers)
user_rating_counts = ratings_with_name.groupby("User-ID").count()["Book-Rating"]
literate_users = user_rating_counts[user_rating_counts > 200].index
filtered_rating = ratings_with_name[ratings_with_name["User-ID"].isin(literate_users)]

# Keep only books that have at least 50 ratings among those engaged users
book_rating_counts = filtered_rating.groupby("Book-Title").count()["Book-Rating"]
famous_books = book_rating_counts[book_rating_counts >= 50].index

final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(famous_books)]

pt = final_ratings.pivot_table(
    index="Book-Title", columns="User-ID", values="Book-Rating"
)
pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

print("\nPivot table shape (books x users):", pt.shape)
print("Similarity matrix shape:", similarity_scores.shape)


def recommend(book_name: str, n: int = 5):
    """Return up to n books most similar to `book_name` (helper for testing)."""
    if book_name not in pt.index:
        return []
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True
    )[1 : n + 1]

    results = []
    for i, score in similar_items:
        title = pt.index[i]
        row = books[books["Book-Title"] == title].drop_duplicates("Book-Title")
        if not row.empty:
            results.append(
                {
                    "title": title,
                    "author": row["Book-Author"].values[0],
                    "image": row["Image-URL-M"].values[0],
                    "score": round(float(score), 3),
                }
            )
    return results


# ---------------------------------------------------------------------------
# 4. Persist everything web.py needs
# ---------------------------------------------------------------------------
pickle.dump(popular_df, open(os.path.join(MODEL_DIR, "popular.pkl"), "wb"))
pickle.dump(pt, open(os.path.join(MODEL_DIR, "pt.pkl"), "wb"))
pickle.dump(
    similarity_scores, open(os.path.join(MODEL_DIR, "similarity_scores.pkl"), "wb")
)
# Save a de-duplicated, lightweight books table (title/author/image only) so
# web.py doesn't need to re-load the full Books.csv at request time
books_lookup = books.drop_duplicates("Book-Title")[
    ["Book-Title", "Book-Author", "Image-URL-M"]
]
pickle.dump(books_lookup, open(os.path.join(MODEL_DIR, "books.pkl"), "wb"))

print(f"\nSaved popular.pkl, pt.pkl, similarity_scores.pkl, books.pkl to {MODEL_DIR}")

if __name__ == "__main__":
    # quick manual test
    sample_title = pt.index[0] if len(pt.index) else None
    if sample_title:
        print(f"\nSample recommendation for '{sample_title}':")
        for r in recommend(sample_title):
            print(" -", r["title"], "by", r["author"], "| score:", r["score"])
