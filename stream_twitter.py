from typing import Dict

import plac
import json
import dataset
import time


from urllib3.exceptions import ProtocolError
from tweet_classifier.TweetScraper import TweetScraper
from tweet_classifier.predictors import TweetPredictor
from tweet_classifier.TweetGeolocator import TweetGeolocator


@plac.annotations(
    auth=plac.Annotation("Path to .json containing twitter auth information: "
                         "\n'consumer_key', 'consumer_secret', 'access_token', 'access_token_secret'. ",
                         kind="option", type=str),
    filter=plac.Annotation(
        "Path to .json containing comma separated values for stream filters: "
        "\n'track', 'languages', 'locations', 'filter_level'. ",
        kind="option", type=str),
    db=plac.Annotation(
        "Name of database with table where tweets will be stored. ",
        kind="option", type=str),
    table=plac.Annotation(
        "Name of table in database where tweets will be stored. ",
        kind="option", type=str)
)
def main(auth="_tweepy_auth.json",
         filter="stream_filter.json",
         db="coronavirus",
         table="live_tweets"):

    # instantiate table
    db = dataset.connect(f"sqlite:///{db}.db")
    table = db.create_table(table, primary_id=False)

    # instantiate predictor
    predictor = TweetPredictor.from_path("tweet_classifier/saved_models/bert_classification/model.tar.gz", "tweet_predictor")

    # instantiate geolocator
    tgl = TweetGeolocator()

    # create on_status_fn
    def on_status_fn(tweet_object: Dict):
        # TODO: geotagging with geotext+geopy
        output_dict = predictor.predict_json(tweet_object)
        tweet_object["prediction"] = output_dict["prediction"]
        tweet_object["prediction_confidence"] = output_dict["prediction_confidence"]
        tweet_object["geolocation"] = tgl.get_geolocation(tweet_object)
        table.insert(tweet_object)


    # instantiate twitter scraper
    ts = TweetScraper.from_json(auth, {"on_status_fn": on_status_fn})

    # instantiate twitter streamer
    with open(filter) as f:
        filter_dict = {k: v.split(",") for k, v in (json.load(f)).items()}
        for k in filter_dict.keys():
            if filter_dict[k] == [""]:
                filter_dict[k] = None
            elif k == 'locations':
                filter_dict['locations'] = list(
                    map(float, filter_dict['locations']))
            elif k == 'filter_level':
                filter_dict['filter_level'] = filter_dict['filter_level'][0]
    while True:
        # Prevent stream stalling due to on_status_fn being slower than stream
        # TODO: use pubsub like redis instead of ignoring errors.
        try:
            ts.stream_tweets(filter_dict)
        except ProtocolError as e:
            print(f"protocol error {time.time()}")
            print(e)
            time.sleep(5)
            continue


if __name__ == '__main__':
    plac.call(main)
