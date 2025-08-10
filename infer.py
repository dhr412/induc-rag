import json, io, uuid
import polars as pl
from datasets import load_dataset
import requests
from groq import Groq
import os
from dotenv import load_dotenv

# -------------------------
# Load datasets and prepare DataFrames
# -------------------------
def _safe_load_json(s):
    if s is None:
        return []
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return []

ds = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
data = ds.to_dict()
df_tmdb = pl.DataFrame(data).with_columns([
    pl.col("popularity").cast(pl.Float64, strict=False)
]).sort("popularity", descending=True).head(100)

pop_min = df_tmdb["popularity"].min()
pop_max = df_tmdb["popularity"].max()
df_tmdb = df_tmdb.with_columns([
    ((pl.col("popularity") - pop_min) / (pop_max - pop_min)).alias("popularity_norm")
])

df_tmdb_filtered = df_tmdb.select([
    pl.col("title").cast(pl.Utf8),
    pl.col("release_date").cast(pl.Utf8),
    pl.col("genres").cast(pl.Utf8),
    pl.col("overview").cast(pl.Utf8),
    pl.col("tagline").cast(pl.Utf8),
    pl.col("popularity_norm").alias("popularity").cast(pl.Float64)
]).with_columns([
    pl.lit(None).cast(pl.Utf8).alias("director")
])

df_tmdb_filtered = df_tmdb_filtered.with_columns([
    pl.col("genres")
      .map_elements(lambda x: ", ".join([g["name"] for g in _safe_load_json(x)]), return_dtype=pl.Utf8)
])

bollywood_csv_url = "https://raw.githubusercontent.com/devensinghbhagtani/Bollywood-Movie-Dataset/main/IMDB-Movie-Dataset(2023-1951).csv"
response = requests.get(bollywood_csv_url)
response.raise_for_status()

df_bollywood = pl.read_csv(io.BytesIO(response.content), ignore_errors=True)
df_bollywood = df_bollywood.with_columns(
    pl.col("year").cast(pl.Int64)
)
df_bollywood_filtered = df_bollywood.filter(
    (pl.col("year") >= 2006) & (pl.col("year") <= 2019)
)

df_bollywood = df_bollywood_filtered.select([
    pl.col("movie_name").alias("title").cast(pl.Utf8),
    pl.col("year").alias("release_date").cast(pl.Utf8),
    pl.col("genre").alias("genres").cast(pl.Utf8),
    pl.col("overview").cast(pl.Utf8),
    pl.col("director").cast(pl.Utf8),
    pl.lit(None).cast(pl.Utf8).alias("tagline"),
    pl.lit(None).cast(pl.Float64).alias("popularity")
]).with_columns([
    pl.lit("Hindi").alias("language")
])
df_tmdb_simple = df_tmdb_filtered.select([
    "title",
    "release_date",
    "genres",
    "overview",
    "director",
    "tagline",
    "popularity"
]).with_columns([
    pl.lit(None).cast(pl.Utf8).alias("director"),
    pl.lit("English").alias("language")
])

df_bollywood_sample = df_bollywood.sample(n=100, seed=24)
df_tmdb_sample = df_tmdb_simple.sample(n=100, seed=24)
combined_df = pl.concat([df_tmdb_sample, df_bollywood_sample]).sample(fraction=1.0, shuffle=True)

# -------------------------
# Session store for movies
# -------------------------
session_movies = {}
session_request_counts = {}

# -------------------------
# Helpers
# -------------------------
def get_or_create_movie(session_id: str):
    if session_id not in session_movies:
        selected_df = combined_df.sample(n=1)
        session_movies[session_id] = selected_df.to_dicts()[0]
    return session_movies[session_id]

def build_facts_and_instruction(movie):
    overview = movie.get("overview") or ""
    release_date = movie.get("release_date") or ""
    popularity = movie.get("popularity") or ""
    runtime = movie.get("runtime") or ""
    vote_average = movie.get("vote_average") or ""
    vote_count = movie.get("vote_count") or ""
    tagline = movie.get("tagline") or ""
    popularity_value = movie.get("popularity") or ""

    genres_list = [g.get("name") for g in _safe_load_json(movie.get("genres")) if isinstance(g, dict)]
    cast_list = [c.get("name") for c in _safe_load_json(movie.get("cast")) if isinstance(c, dict)]
    main_cast = cast_list[:6] if cast_list else []
    crew = _safe_load_json(movie.get("crew"))
    directors = [c.get("name") for c in crew if isinstance(c, dict) and c.get("job") and c.get("job").lower() == "director"]
    directors = directors or []

    facts = [
        f"Overview: {overview or 'N/A'}",
        f"Tagline: {tagline or 'N/A'}",
        f"Release date: {release_date or 'N/A'}",
        f"Genres: {', '.join(genres_list) if genres_list else (movie.get('genres') or 'N/A')}",
        f"Main cast (top billed): {', '.join(main_cast) if main_cast else 'N/A'}",
        f"Director(s): {', '.join(directors) if directors else (movie.get('director') or 'N/A')}",
        f"Runtime (minutes): {runtime or 'N/A'}",
        f"Average rating: {vote_average or 'N/A'} (votes: {vote_count or 'N/A'})",
        f"Popularity score (range 0–1): {f'{popularity_value:.3f}' if isinstance(popularity_value, (int, float)) else 'N/A'}",
        f"Language: {movie.get('language') or 'N/A'}"
    ]

    facts_block = "\n".join(facts)

    system_instruction = f"""
You are an assistant that answers only 'Yes' or 'No' about a hidden movie
based on the factual information provided in the 'Movie facts' section below.

RULES:
1. If the user explicitly guesses the movie (e.g., "is the movie X?" or "is it X?"),
   compare their guess (case-insensitive) to the hidden title and hidden franchise.
2. If the guess matches exactly or clearly refers to the correct franchise, respond:
   "Yes, that is correct! The movie is {movie.get("title").capitalize()}."
3. If the guess is incorrect, respond:
   "No, that is not the movie or its franchise."
4. For other questions that can be answered with the provided facts, respond only with "Yes" or "No".
5. For questions about popularity:
   - The popularity score is normalized between 0 (least popular) and 1 (most popular).
   - If score ≥ 0.66, consider the movie "very popular".
   - If 0.33 ≤ score < 0.66, consider it "moderately popular".
   - If score < 0.33, consider it "less popular".
   - Use this mapping to answer "Yes" or "No" for popularity-related questions.
6. Never provide extra explanations or reveal the title unless the user explicitly asks or guesses correctly.

The movie title is: "{movie.get("title").lower()}"
The movie language is: "{movie.get("language").lower()}"
"""
    return facts_block, system_instruction

# -------------------------
# LLM client
# -------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------
# Public functions
# -------------------------
def ask_question(user_question: str, session_id: str):
    session_request_counts[session_id] = session_request_counts.get(session_id, 0) + 1
    movie = get_or_create_movie(session_id)
    if session_request_counts.get(session_id, 0) > 10:
        return f"Game over! The movie was {movie.get('title')}"
    facts_block, system_instruction = build_facts_and_instruction(movie)

    prompt = (
        system_instruction + "\n\n"
        "Movie facts:\n"
        f"{facts_block}\n\n"
        f"User question: {user_question.strip().lower()}"
    )

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_completion_tokens=50,
        top_p=1,
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()

def get_hint(session_id: str):
    movie = get_or_create_movie(session_id)
    facts_block, _ = build_facts_and_instruction(movie)

    hint_prompt = f"""
You are an assistant helping someone guess a hidden movie based on the following factual information. 
Please provide a single, short, subtle hint (like a famous quote, tagline, or interesting clue) that could help guess the movie, but do NOT reveal the movie title or any explicit spoilers.

Movie facts:
{facts_block}

Hint:
"""
    hint_completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": hint_prompt}],
        temperature=0.7,
        max_completion_tokens=50,
        top_p=1,
        stream=False,
        stop=None
    )
    return hint_completion.choices[0].message.content.strip()
