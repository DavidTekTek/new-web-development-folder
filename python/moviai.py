import sys
import time
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore, Style


init(autoreset=True)


def load_data(file_path: str = "imdb_top_1000.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)

        required_cols = {"Series_Title", "Genre", "Overview", "IMDB_Rating"}
        missing = required_cols - set(df.columns)
        if missing:
            print(Fore.RED + f"Error: Dataset is missing columns: {sorted(missing)}")
            sys.exit(1)

        df["combined_features"] = df["Genre"].fillna("") + " " + df["Overview"].fillna("")
        return df

    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        print(Fore.YELLOW + "Tip: Put 'imdb_top_1000.csv' in the same folder as main.py.")
        sys.exit(1)


movies_df = load_data()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def list_genres(df: pd.DataFrame):
    return sorted(
        set(
            genre.strip()
            for sublist in df["Genre"].dropna().astype(str).str.split(", ")
            for genre in sublist
        )
    )


genres = list_genres(movies_df)


def display_recommendations(recs, name: str):
    print(Fore.YELLOW + f"\n🍿 AI-Analyzed Movie Recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
        print(f"{Fore.CYAN}{idx}. 🎬 {title} (Polarity: {polarity:.2f}, {sentiment})")


def processing_animation():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)
    print()


def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df

    if genre:
        filtered_df = filtered_df[filtered_df["Genre"].str.contains(genre, case=False, na=False)]

    if rating is not None:
        filtered_df = filtered_df[filtered_df["IMDB_Rating"] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    recommendations = []
    mood_polarity = None
    if mood:
        mood_polarity = TextBlob(mood).sentiment.polarity

    for _, row in filtered_df.iterrows():
        overview = row["Overview"]
        if pd.isna(overview):
            continue

        polarity = TextBlob(str(overview)).sentiment.polarity

        if mood_polarity is None:
            ok = True
        else:
            if mood_polarity < 0:
                ok = polarity > 0
            else:
                ok = polarity >= 0

        if ok:
            recommendations.append((row["Series_Title"], polarity))

        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendations found."


def handle_ai(name: str):
    print(Fore.BLUE + "\n🔎 Let's find the perfect movie for you!\n")

    # Show genres (numbered)
    print(Fore.GREEN + "Available Genres:")
    for idx, genre in enumerate(genres, 1):
        print(f"{Fore.CYAN}{idx}. {genre}")
    print()

    # Pick genre
    while True:
        genre_input = input(Fore.YELLOW + "Enter genre number or name: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print(Fore.RED + "Invalid input. Try again.\n")

    # Mood
    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()

    # Analyze mood
    print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
    processing_animation()
    polarity = TextBlob(mood).sentiment.polarity
    mood_desc = "positive 😊" if polarity > 0 else "negative 😞" if polarity < 0 else "neutral 😐"
    print(f"{Fore.GREEN}Your mood is {mood_desc} (Polarity: {polarity:.2f}).\n")

    # Rating (optional)
    while True:
        rating_input = input(Fore.YELLOW + "Enter minimum IMDB rating (7.6–9.3) or 'skip': ").strip()
        if rating_input.lower() == "skip":
            rating = None
            break
        try:
            rating = float(rating_input)
            if 0.0 <= rating <= 10.0:
                break
            print(Fore.RED + "Rating out of range (0–10). Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")

    # Find movies
    print(f"{Fore.BLUE}\nFinding movies for {name}", end="", flush=True)
    processing_animation()

    recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
    if isinstance(recs, str):
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)

    return genre, mood, rating


def main():
    print(Fore.BLUE + "🤖 Welcome to your Personal Movie Recommendation Assistant! 🤖\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()

    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    genre, mood, rating = handle_ai(name)

    while True:
        action = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if action == "no":
            print(Fore.GREEN + f"\nEnjoy your movie picks, {name}! 🎬🍿\n")
            break
        elif action == "yes":
            recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
            if isinstance(recs, str):
                print(Fore.RED + recs + "\n")
            else:
                display_recommendations(recs, name)
        else:
            print(Fore.RED + "Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
