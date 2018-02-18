from TwitterAPI import TwitterAPI

SEARCH_TERM = 'pizza'

CONSUMER_KEY = 'V0TldDqji7JVVrJSTQwFLZ8Ue'
CONSUMER_SECRET = 'RH9hn9bWgfJXwrGdUeeR2oVx1VsFAS8odASLGRDaFJQGrKtBnn'
ACCESS_TOKEN_KEY = '962614749409173504-ceErO3PaktkBPRg7v4ROhD4gvAvdNUZ'
ACCESS_TOKEN_SECRET = 'ybXGN8ZJkkmNJR78nTuj0p2IHvXqbryrtCSXuK37YoVpe'

api = TwitterAPI(CONSUMER_KEY,
                 CONSUMER_SECRET,
                 ACCESS_TOKEN_KEY,
                 ACCESS_TOKEN_SECRET)

r = api.request('search/tweets', {'q': SEARCH_TERM})

for item in r:
    print(item['text'] if 'text' in item else item)

print('\nQUOTA: %s' % r.get_rest_quota())
