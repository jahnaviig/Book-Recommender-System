

import os
import pickle

from flask import Flask, render_template, request, flash, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-in-production"  # needed for flash messages

# ---------------------------------------------------------------------------
# Load model artifacts once, at startup
# ---------------------------------------------------------------------------
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


popular_df = load_pickle("popular.pkl")

# These two are optional: only present if train_model.py has been run against
# the full dataset. The app still works (popularity page only) without them.
try:
    pt = load_pickle("pt.pkl")
    similarity_scores = load_pickle("similarity_scores.pkl")
    books_lookup = load_pickle("books.pkl")
    RECOMMEND_READY = True
except FileNotFoundError:
    pt = None
    similarity_scores = None
    books_lookup = None
    RECOMMEND_READY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_similar_books(book_title, n=8):
    """Return up to n books similar to book_title using the cosine
    similarity matrix built in train_model.py. Matching is case-insensitive
    and tolerant of partial titles."""
    if not RECOMMEND_READY:
        return None  # signals "feature unavailable" to the route

    titles = list(pt.index)
    lower_titles = [t.lower() for t in titles]
    query = book_title.strip().lower()

    # exact match first, then "contains" match
    if query in lower_titles:
        match_idx = lower_titles.index(query)
    else:
        candidates = [i for i, t in enumerate(lower_titles) if query in t]
        if not candidates:
            return []
        match_idx = candidates[0]

    matched_title = titles[match_idx]
    similar_items = sorted(
        list(enumerate(similarity_scores[match_idx])),
        key=lambda x: x[1],
        reverse=True,
    )[1 : n + 1]

    results = []
    for idx, score in similar_items:
        title = pt.index[idx]
        row = books_lookup[books_lookup["Book-Title"] == title]
        if row.empty:
            continue
        results.append(
            {
                "title": title,
                "author": row["Book-Author"].values[0],
                "image": row["Image-URL-M"].values[0],
            }
        )

    return {"matched_title": matched_title, "results": results}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_ratings"].values),
        rating=list(popular_df["avg_rating"].values),
    )


@app.route("/recommend", methods=["GET", "POST"])
def recommend_ui():
    searched_title = None
    results = []
    no_match = False
    feature_disabled = not RECOMMEND_READY

    if request.method == "POST":
        searched_title = request.form.get("book_title", "").strip()
        if searched_title and RECOMMEND_READY:
            outcome = get_similar_books(searched_title)
            if outcome is None:
                feature_disabled = True
            elif len(outcome["results"]) == 0:
                no_match = True
            else:
                searched_title = outcome["matched_title"]
                results = outcome["results"]

    return render_template(
        "recommend.html",
        searched_title=searched_title,
        results=results,
        no_match=no_match,
        feature_disabled=feature_disabled,
    )


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please fill in all fields before sending.", "error")
        else:
            # In a real deployment you'd email this or save it to a database.
            # For now we just log it server-side and confirm to the user.
            print(f"[Contact form] {name} <{email}>: {message}")
            flash("Thanks for reaching out! Your message has been received.", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
