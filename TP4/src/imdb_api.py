from dotenv import load_dotenv
import requests
import os
from dataclasses import dataclass
import json
from typing import List, Dict


@dataclass
class ApiResponse:
    Title: str  # "Guardians of the Galaxy Vol. 2"
    Released: str  # "05 May 2017"
    Runtime: str  # "136 min"
    Genre: str  # "Action, Adventure, Comedy, Sci-Fi"
    Plot: str  # "The Guardians struggle to keep together..."
    Language: str  # "English, Mandarin, Russian"
    Country: str  # "United States"
    Ratings: List[
        Dict[str, str]
    ]  # [{"Source": "Internet Movie Database", "Value": "7.6/10"}, ...]
    MetaScore: str  # "67"
    imdbRating: str  # "7.6"
    imdbVotes: str  # "610,832"
    imdbID: str  # "tt3896198"
    BoxOffice: str


load_dotenv()
API_KEY = os.getenv("OMDB_API_KEY")
API_URL = "http://www.omdbapi.com/"


def get_movie_info(imdb_id: str) -> dict:
    res = requests.get(API_URL, {"apikey": API_KEY, "i": imdb_id})
    res.raise_for_status()

    return res.json()


print(json.dumps(get_movie_info("tt3896198"), indent=4))
