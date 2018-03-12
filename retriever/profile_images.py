import yaml
import pandas as pd
import urllib.request as req
import urllib.error
import sys
import pickle

if __name__ == '__main__':
    conf_file = open('./conf/gender_train.yaml')
    gender_conf = yaml.load(conf_file)
    print(gender_conf)

    text_conf = gender_conf['text']
    data = pd.read_csv(text_conf['path'], encoding='latin1')

    profile_images = {}
    ok = 0
    not_found = 0
    for uid, url in zip(data['_unit_id'], data['profileimage']):
        try:
            response = req.urlopen(url)
            img = response.read()
            profile_images[uid] = img
            ok += 1
        except urllib.error.URLError as e:
            not_found += 1

        sys.stdout.write('ok = %d, not found = %d \r' % (ok, not_found))
        sys.stdout.flush()
        break

    pickle.dump(profile_images, open('./data/profile_images', 'wb'))
