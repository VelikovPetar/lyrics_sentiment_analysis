import requests

BASE_URL = 'https://api.lyrics.ovh/v1/artist/title'


def get_lyrics(artist, title):
    endpoint = BASE_URL.replace('artist', artist).replace('title', title)
    response = requests.get(endpoint)
    print(response.text)
    if response.status_code == 200:
        return (True, response.json())
    else:
        return (False, None if response is None else response.json())
