from typing import Tuple, Dict

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
        res = self.es_conn.filter("term", feature_code="PCLI").query(q)[:5].execute()
        if len(res) > 0:
            coordinates = [float(x) for x in res[0].coordinates.split(",")]
            return coordinates[0], coordinates[1]
        else:
            # try location matching
            print("No country found: ", country)
            return self._get_location_coordinates(country)

    @lru_cache(None)
    def _get_location_coordinates(self, location: str) -> Tuple[float, float]:
        q = {"multi_match":
                 {"query": location,
                  "fields": ['name^5', 'asciiname^5', 'alternativenames'],
                  "type": "phrase"}}
        res = self.es_conn.query(q)[:5].execute()
        if len(res) > 0:
            coordinates = [float(x) for x in res[0].coordinates.split(",")]
            return coordinates[0], coordinates[1]
        else:
            print("No location found: ", location)
            return 'None'

    def get_geolocation(self, tweet_object: Dict):
        text = tweet_object["text"]
        if tweet_object["coordinates"] != 'None':
            return tweet_object["coordinates"]
        else:
            # get geolocation from place
            if tweet_object["place"] != 'None':
                try:
                    place = json.loads(tweet_object["place"].replace("'", '"'))
                    shp = shape(place["bounding_box"])
                    x, y = self._random_point_in_shp(shp)
                    return str({'type': 'Point', 'coordinates': [x, y]})
                except Exception as e:
                    print(
                        f"Error while parsing geolocation from place: {e.message}")
                    pass

            # get geolocation from tweet
            try:
                text = text.replace("#", "")
                text = self.camel_to_ws.sub(' ', text)
                places = GeoText(text)
                coordinates = 'None'
                if places.cities:
                    # choose first city by default
                    place = places.cities[0]
                    coordinates = self._get_location_coordinates(place)
                elif places.countries:
                    # choose first country if no city
                    place = places.countries[0]
                    coordinates = self._get_country_coordinates(place)
                if coordinates != 'None':
                    x = coordinates[0]
                    y = coordinates[1]
                    return str({'type': 'Point', 'coordinates': [x, y]})
            except Exception as e:
                print(f"Error while parsing geolocation from text: {e.message}")
                pass

            # get geolocation from user location
            if tweet_object["user_location"]:
                try:
                    places = GeoText(tweet_object["user_location"])
                    coordinates = 'None'
                    if places.cities:
                        # choose first city by default
                        place = places.cities[0]
                        coordinates = self._get_location_coordinates(place)
                    elif places.countries:
                        # choose first country if no city
                        place = places.countries[0]
                        coordinates = self._get_country_coordinates(place)
                    if coordinates != 'None':
                        x = coordinates[0]
                        y = coordinates[1]
                        return str({'type': 'Point', 'coordinates': [x, y]})
                except Exception as e:
                    print(
                        f"Error while parsing geolocation from user location: {e.message}")
                    pass

            return 'None'




