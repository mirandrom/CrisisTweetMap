from typing import Tuple, Dict, Optional, List

import numpy as np
import re
import json
from functools import lru_cache

from geotext import GeoText
from shapely.geometry import Point, Polygon, shape
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

class TweetGeolocator:
    def __init__(self):
        self.es_conn = self._get_es_conn()

        self.camel_to_ws = re.compile(r'(?<!^)(?=[A-Z])')

    def _get_es_conn(self):
        try:
            kwargs = dict(
                hosts=['localhost'],
                port=9200,
                use_ssl=False,
            )
            CLIENT = Elasticsearch(**kwargs)
            es_conn = Search(using=CLIENT, index="geonames")
            es_conn.count()
            return es_conn
        except:
            raise ConnectionError("Error establishing connection with ES container")

    def _random_point_in_shp(self, shp: Polygon):
        within = False
        while not within:
            x = np.random.uniform(shp.bounds[0], shp.bounds[2])
            y = np.random.uniform(shp.bounds[1], shp.bounds[3])
            within = shp.contains(Point(x, y))
        return x, y

    @lru_cache(None)
    def _get_country_coordinates(self, country: str) -> Tuple[float, float]:
        q = {"multi_match":
                 {"query": country,
                  "fields": ['name', 'asciiname', 'alternativenames'],
                  "type": "phrase"}
             }
        results = self.es_conn.filter("term", feature_code="PCLI").query(q)[:5].execute()
        if len(results) > 0:
            coordinates = [tuple([float(x) for x in res.coordinates.split(",")][:2]) for res in results]
            return coordinates
        else:
            # try location matching
            print("No country found: ", country)
            return self._get_location_coordinates(country)

    @lru_cache(None)
    def _get_location_coordinates(self, location: str) -> Optional[List[Tuple[float, float]]]:
        q = {"multi_match":
                 {"query": location,
                  "fields": ['name^5', 'asciiname^5', 'alternativenames'],
                  "type": "phrase"}}
        results = self.es_conn.query(q)[:5].execute()
        if len(results) > 0:
            coordinates = [tuple([float(x) for x in res.coordinates.split(",")][:2]) for res in results]
            return coordinates
        else:
            print("No location found: ", location)
            return None

    def get_geolocation(self, tweet_object: Dict):
        text = tweet_object["text"]
        locations = {"tweet": [], "place": [],
                     "content_cities": [], "content_countries": [],
                     "user_cities": [], "user_countries": []}
        if tweet_object["coordinates"] != 'None':
            locations ["tweet"] += [json.loads(tweet_object["coordinates"].replace("'", '"'))["coordinates"]]
        else:
            # get geolocation from place
            if tweet_object["place"] != 'None':
                try:
                    place = json.loads(tweet_object["place"].replace("'", '"'))
                    shp = shape(place["bounding_box"])
                    x, y = self._random_point_in_shp(shp)
                    locations["place"] += [[x,y]]
                except Exception as e:
                    print(f"Error while parsing geolocation from place: {e}")
                    pass

            # get geolocation from tweet
            try:
                text = text.replace("#", "")
                text = self.camel_to_ws.sub(' ', text)
                places = GeoText(text)
                if places.cities:
                    for place in places.cities:
                        coordinates = self._get_location_coordinates(place)
                        if coordinates:
                            locations["content_cities"] += coordinates
                if places.countries:
                    for place in places.countries:
                        coordinates = self._get_country_coordinates(place)
                        if coordinates:
                            locations["content_countries"] += coordinates
            except Exception as e:
                print(f"Error while parsing geolocation from text: {e}")
                pass

            # get geolocation from user location
            if tweet_object["user_location"]:
                try:
                    places = GeoText(tweet_object["user_location"])
                    if places.cities:
                        for place in places.cities:
                            coordinates = self._get_location_coordinates(place)
                            if coordinates:
                                locations["user_cities"] += coordinates
                    if places.countries:
                        locations["user_countries"] = []
                        for place in places.countries:
                            coordinates = self._get_country_coordinates(place)
                            if coordinates:
                                locations["user_countries"] += coordinates
                except Exception as e:
                    print(
                        f"Error while parsing geolocation from user location: {e}")
                    pass

        return locations
