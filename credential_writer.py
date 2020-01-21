import json

import pandas as pd
from pandas import DataFrame

from twitterServant import TwitterAnalyzer, preProcessDataAndExtractHashtags

# Use this method to create the appropriate credential file (or create the json by hand..)
def write_your_Credentials():
    credentials = {}
    credentials['CONSUMER_KEY'] = 'INSERT YOUR CONSUMER KEY'
    credentials['CONSUMER_SECRET'] = 'INSERT YOUR CONSUMER SECRET '
    credentials['ACCESS_TOKEN'] = 'INSERT YOUR ACESS TOKEN'
    credentials['ACCESS_SECRET'] = 'INSERT YOU ACCESS SECRET KEY'

    # Save the credentials object to file
    with open("twitter_credentials.json", "w") as file:
        json.dump(credentials, file)




if __name__ == '__main__':
    write_your_Credentials()





