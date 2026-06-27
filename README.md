## Shelfwise — A Book Recommender System

A Flask web app that recommends books two ways:

1. **Popularity-based** — shows the top 50 books by average rating
2. **Collaborative filtering** — type a book you like, get similar books based on what other readers with similar taste enjoyed

Built with Python, Flask, pandas, and scikit-learn (cosine similarity).

### Pages
- **Home** — Top 50 rated books
- **Recommend** — Search a title, get similar book suggestions
- **Contact** — Get in touch form

### Tech stack
- Flask (backend + routing)
- pandas / numpy (data processing)
- scikit-learn (cosine similarity for recommendations)
- HTML/CSS (Jinja templates)

### Running locally
```bash
pip install -r requirements.txt
python web.py
```
Then open `http://127.0.0.1:5000`
