from dotenv import load_dotenv
import requests
import os
from dataclasses import dataclass, asdict
import json
from typing import List, Dict, Union
from datetime import datetime, date
from abc import ABC, abstractmethod

load_dotenv()


@dataclass
class Movie:
    budget: int
    genres: str
    imdb_id: str
    original_title: str
    overview: str
    popularity: float
    production_companies: int
    production_countries: int
    release_date: date
    revenue: int
    runtime: int
    spoken_languages: int
    vote_average: float
    vote_count: int


class ApiAdapter(ABC):

    @abstractmethod
    def query_api(self, imdb_id: str) -> Dict:
        pass

    @abstractmethod
    def parse_api_data(self, api_data: Dict) -> Movie:
        pass

    def get_movie_info(self, imdb_id: str) -> Movie:
        return self.parse_api_data(self.query_api(imdb_id))


class TMDB_API(ApiAdapter):
    EXTERNAL_ID_URL = "https://api.themoviedb.org/3/find/"
    MOVIE_URL= "https://api.themoviedb.org/3/movie/"
    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('TMDB_READ_ACCESS_TOKEN')}",
    }

    @dataclass
    class ApiResponse:
        budget: int # 200000000
        genres: List[Dict[str, Union[str, int]]] # [{"id": 28, "name": "Action"}, ...]
        imdb_id: str # "tt3896198"
        origin_country: List[str] # ["US"]
        original_language: str # "en"
        original_title: str # "Guardians of the Galaxy Vol. 2"
        overview: str # "The Guardians struggle to keep together..."
        popularity: float # 30.0
        production_companies: List[Dict[str, Union[str, int]]] # [{"id": 420, "name": "Marvel Studios"}, ...]
        production_countries: List[Dict[str, str]] # [{"iso_3166_1": "US", "name": "United States of America"}, ...]
        release_date: str # "2017-04-19"
        revenue: int # 863416141
        runtime: int # 137
        spoken_languages: List[Dict[str, str]] # [{"iso_639_1": "en", "name": "English"}, ...]
        vote_average: float # 7.6
        vote_count: int # 6108

    def query_api(self, imdb_id: str) -> Dict:
        res = requests.get(
            self.EXTERNAL_ID_URL + imdb_id,
            {"external_source": "imdb_id"},
            headers=self.HEADERS
        )
        res.raise_for_status()
        movie_id = res.json()["movie_results"][0]["id"]
        if movie_id is None:
            raise Exception("No movie ID found")
        res = requests.get(
            self.MOVIE_URL + str(movie_id),
            headers=self.HEADERS
        )
        res.raise_for_status()
        return res.json()
    
    def parse_api_data(self, api_data: Dict) -> Movie:
        response = self.ApiResponse(
            **{
                k: v
                for k, v in api_data.items()
                if k in self.ApiResponse.__annotations__.keys()
            }
        )
        return Movie(
            budget=response.budget,
            genres=response.genres[0].get('name', None),
            imdb_id=response.imdb_id,
            original_title=response.original_title,
            overview=response.overview,
            popularity=response.popularity,
            production_companies=len(response.production_companies),
            production_countries=len(response.production_countries),
            release_date=datetime.strptime(response.release_date, "%Y-%m-%d").date(),
            revenue=response.revenue,
            runtime=response.runtime,
            spoken_languages=len(response.spoken_languages),
            vote_average=response.vote_average,
            vote_count=response.vote_count,
        )

class OMDB_API(ApiAdapter):
    API_KEY = os.getenv("OMDB_API_KEY")
    API_URL = "http://www.omdbapi.com/"

    @dataclass
    class ApiResponse:
        Title: str  # "Guardians of the Galaxy Vol. 2"
        Released: str  # "05 May 2017"
        Runtime: str  # "136 min"
        Genre: str  # "Action, Adventure, Comedy, Sci-Fi"
        Plot: str  # "The Guardians struggle to keep together..."
        Language: str  # "English, Mandarin, Russian"
        Country: str  # "United States, Italy, Germany, France"
        Ratings: List[
            Dict[str, str]
        ]  # [{"Source": "Internet Movie Database", "Value": "7.6/10"}, ...]
        Metascore: str  # "67"
        imdbRating: str  # "7.6"
        imdbVotes: str  # "610,832"
        imdbID: str  # "tt3896198"
        BoxOffice: str  # "$13,681,765"

    def query_api(self, imdb_id: str) -> Dict:
        res = requests.get(self.API_URL, {"apikey": self.API_KEY, "i": imdb_id})
        res.raise_for_status()
        return res.json()

    def parse_api_data(self, api_data: Dict) -> Movie:
        response = self.ApiResponse(
            **{
                k: v
                for k, v in api_data.items()
                if k in self.ApiResponse.__annotations__.keys()
            }
        )
        def parse_release_date(date: str) -> date:
            return datetime.strptime(date, "%d %b %Y")
        return Movie(
            budget=None,
            genres=response.Genre.split(", ")[
                0
            ],  # Search for the first coincidence with the existing genres
            imdb_id=response.imdbID,
            original_title=response.Title,
            overview=response.Plot,
            popularity=None,  # TODO: Calculate the popularity based on the imdbVotes, adjusting the ranges
            production_companies=None,  # Cannot be computed
            production_countries=len(response.Country.split(", ")),
            release_date=parse_release_date(response.Released),
            revenue=int(response.BoxOffice.replace("$", "").replace(",", "")),
            runtime=int(response.Runtime.split()[0]),
            spoken_languages=len(response.Language.split(", ")),
            vote_average=float(response.imdbRating),
            vote_count=int(response.imdbVotes.replace(",", "")),
        )
