import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# =============== NEW: folder for plots =================
PLOTS_DIR = "eda_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
# =======================================================

books = pd.read_csv("dataset/books_with_moods.csv")
ratings = pd.read_csv("dataset/ratings.csv")

print("Books shape:", books.shape)
print("Ratings shape:", ratings.shape)

import re


def split_genres(genre_str):
    if pd.isna(genre_str):
        return []

    # Convert string list → actual Python list
    # Example: "['fiction', 'fantasy']" → ['fiction', 'fantasy']
    try:
        genres = eval(genre_str)
        return [g.strip() for g in genres]
    except:
        # fallback if eval fails
        cleaned = re.sub(r"[\[\]']", "", genre_str)  # remove brackets & quotes
        return [g.strip() for g in cleaned.split(",") if g.strip()]


# =============================================================
# BOOKS EDA
# =============================================================

# 1. Description length histogram
books["desc_len"] = books["description"].fillna("").apply(
    lambda x: len(str(x).split())
)

plt.figure(figsize=(8, 5))
plt.hist(books["desc_len"], bins=50)
plt.title("Description Length Distribution (Words)")
plt.xlabel("Word Count")
plt.ylabel("Book Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "description_length_histogram.png"))
plt.close()

print("Description Length Stats:")
print(books["desc_len"].describe())

# 2. Genre frequency bar chart
books["genre_list"] = books["genres"].apply(split_genres)
genre_exploded = books.explode("genre_list")
genre_exploded = genre_exploded[genre_exploded["genre_list"].notna()]

genre_counts = genre_exploded["genre_list"].value_counts()

plt.figure(figsize=(10, 6))
genre_counts.head(20).plot(kind="bar")
plt.title("Top 20 Genres")
plt.xlabel("Genre")
plt.ylabel("Book Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top20_genres_bar.png"))
plt.close()

print("Top 20 genres:")
print(genre_counts.head(20))

# 3. Primary mood frequency bar
mood_counts = books["mood_primary"].value_counts()

plt.figure(figsize=(8, 5))
mood_counts.plot(kind="bar")
plt.title("Primary Mood Frequency")
plt.xlabel("Mood")
plt.ylabel("Book Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "primary_mood_frequency_bar.png"))
plt.close()

# 3b. Primary mood pie chart
plt.figure(figsize=(6, 6))
mood_counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Primary Mood Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "primary_mood_distribution_pie.png"))
plt.close()

print("Primary Mood Counts:")
print(mood_counts)

# 4. Mood primary score distribution
plt.figure(figsize=(8, 5))
plt.hist(books["mood_primary_score"].dropna(), bins=30)
plt.title("mood_primary_score Distribution")
plt.xlabel("Score")
plt.ylabel("Book Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mood_primary_score_distribution.png"))
plt.close()

print("Mood Primary Score Stats:")
print(books["mood_primary_score"].describe())

# =============================================================
# RATINGS EDA
# =============================================================

# 5. Rating distribution histogram
plt.figure(figsize=(8, 5))
plt.hist(ratings["rating"], bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth=0.8)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks([1,2,3,4,5])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rating_distribution_histogram.png"))
plt.close()

print("Rating Stats:")
print(ratings["rating"].describe())

# 6. Ratings per user distribution
ratings_per_user = ratings.groupby("user_id")["book_id"].count()

plt.figure(figsize=(8, 5))
plt.hist(ratings_per_user, bins=50)
plt.title("Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("User Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ratings_per_user_histogram.png"))
plt.close()

print("Ratings per User Stats:")
print(ratings_per_user.describe())

# 7. Ratings per book distribution
ratings_per_book = ratings.groupby("book_id")["user_id"].count()

plt.figure(figsize=(8, 5))
plt.hist(ratings_per_book, bins=50)
plt.title("Ratings per Book")
plt.xlabel("Number of Ratings")
plt.ylabel("Book Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ratings_per_book_histogram.png"))
plt.close()

print("Ratings per Book Stats:")
print(ratings_per_book.describe())

# =============================================================
# CROSS EDA — Genre × Mood Heatmap
# =============================================================

cross_df = books.copy()
# cross_df["genre_list"] = cross_df["genres"].apply(split_genres)
# cross_exploded = cross_df.explode("genre_list")
#
# cross_exploded = cross_exploded[
#     (cross_exploded["genre_list"].notna()) &
#     (cross_exploded["mood_primary"].notna())
# ]
#
# genre_mood_matrix = pd.crosstab(
#     cross_exploded["genre_list"],
#     cross_exploded["mood_primary"]
# )
#
# plt.figure(figsize=(14, 8))
# sns.heatmap(genre_mood_matrix, cmap="viridis")
# plt.title("Genre × Mood Heatmap")
# plt.xlabel("Mood")
# plt.ylabel("Genre")
# plt.tight_layout()
# plt.savefig(os.path.join(PLOTS_DIR, "genre_mood_heatmap.png"))
# plt.close()
#
# print("Genre × Mood Table (Top Rows):")
# print(genre_mood_matrix.head(10))


books["genre_list"] = books["genres"].apply(split_genres)
df = books.explode("genre_list")
df = df[df["genre_list"].notna() & df["mood_primary"].notna()]

# Count matrix
genre_mood = pd.crosstab(df["genre_list"], df["mood_primary"])

# Top 20 genres
top_genres = genre_mood.sum(axis=1).sort_values(ascending=False).head(20).index
top = genre_mood.loc[top_genres]

# Normalize by row (percentage)
top_norm = top.div(top.sum(axis=1), axis=0)

# Sort columns alphabetically or by importance
top_norm = top_norm[top_norm.sum().sort_values(ascending=False).index]

plt.figure(figsize=(14, 10))
sns.heatmap(
    top_norm,
    cmap="magma",
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Proportion of Books"}
)

plt.title("Genre × Mood Heatmap (Top 20, Normalized)", fontsize=16)
plt.xlabel("Mood")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "genre_mood_heatmap.png"))
plt.show()
