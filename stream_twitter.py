import plac
import json
from tweet_scraper.TweetScraper import TweetScraper


@plac.annotations(
    auth=plac.Annotation("Path to .json containing twitter auth information: "
                         "\n'consumer_key', 'consumer_secret', 'access_token', 'access_token_secret'. ",
                         kind="option", type=str),
    filter=plac.Annotation("Path to .json containing comma separated values for stream filters: "
                           "\n'track', 'languages', 'locations', 'filter_level'. ",
                           kind="option", type=str),
    db=plac.Annotation("Name of database with table where tweets will be stored. ",
                       kind="option", type=str),
    table=plac.Annotation("Name of table in database where tweets will be stored. ",
                          kind="option", type=str)
)
def main(auth="_tweepy_auth.json",
         filter="stream_filter.json",
         db="flood_test",
         table="live_tweets"):
    # instantiate twitter scraper
    ts = TweetScraper.from_json(auth, dict(db=db, table=table))

    # instantiate twitter streamer
    with open(filter) as f:
        filter_dict = {k:v.split(",") for k,v in (json.load(f)).items()}
        for k in filter_dict.keys():
            if filter_dict[k] == [""]:
                filter_dict[k] = None
            elif k == 'locations':
                filter_dict['locations'] = list(map(float, filter_dict['locations']))
            elif k == 'filter_level':
                filter_dict['filter_level'] = filter_dict['filter_level'][0]
    ts.stream_tweets(filter_dict)


if __name__ == '__main__':
    plac.call(main)