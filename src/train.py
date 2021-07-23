import spotipy
from spotipy import SpotifyClientCredentials, util
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, r2_score, roc_auc_score

from model import Cluster

os.environ["SPOTIPY_CLIENT_ID"] = "462608ee0dd145a3bd9eb93ec19c257e"
os.environ["SPOTIPY_CLIENT_SECRET"] = "517667e8f5c14015b331db57d3f85f56"

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlists = sp.user_playlists('spotify')
while playlists:
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None


bands = ['Spiritbox','Mestis','Lorna Shore']

result = []

for i in bands:
  name = i
  search_band = sp.search(name, market='CO')
  search_band = search_band['tracks']['items'][1]['artists']
  result.append(search_band)


artist_uri = []

for i in range(len(result)):
  uri = result[i][0]['uri']
  artist_uri.append(uri)


sp_albums = []

for uri in artist_uri:
  each_album = sp.artist_albums(uri, album_type='album')
  sp_albums.append(each_album)


album_names = []
album_uris = []
album_artist = []

for i in range(len(sp_albums)):
  for j in range(len(sp_albums[i]['items'])):
    album_names.append(sp_albums[i]['items'][j]['name'])
    album_uris.append(sp_albums[i]['items'][j]['uri'])
    album_artist.append(sp_albums[i]['items'][j]['artists'][0]['name'])


def albumSongs(uri):

  album = uri 

  spotify_albums[album] = {} 

  spotify_albums[album]['artist'] = []
  spotify_albums[album]['album'] = [] 
  spotify_albums[album]['track_number'] = []
  spotify_albums[album]['id'] = []
  spotify_albums[album]['name'] = []
  spotify_albums[album]['uri'] = []

  tracks = sp.album_tracks(album) 
  
  for n in range(len(tracks['items'])): 
    spotify_albums[album]['artist'].append(album_artist[album_count])
    spotify_albums[album]['album'].append(album_names[album_count]) 
    spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
    spotify_albums[album]['id'].append(tracks['items'][n]['id'])
    spotify_albums[album]['name'].append(tracks['items'][n]['name'])
    spotify_albums[album]['uri'].append(tracks['items'][n]['uri']) 


spotify_albums = {}
album_count = 0

for i in album_uris: 
    albumSongs(i)
    #print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary")
    album_count+=1

def audio_features(album):

    spotify_albums[album]['acousticness'] = []
    spotify_albums[album]['danceability'] = []
    spotify_albums[album]['energy'] = []
    spotify_albums[album]['instrumentalness'] = []
    spotify_albums[album]['liveness'] = []
    spotify_albums[album]['loudness'] = []
    spotify_albums[album]['speechiness'] = []
    spotify_albums[album]['tempo'] = []
    spotify_albums[album]['valence'] = []
    spotify_albums[album]['popularity'] = []

    track_count = 0
    for track in spotify_albums[album]['uri']:

        features = sp.audio_features(track)
        
        spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
        spotify_albums[album]['danceability'].append(features[0]['danceability'])
        spotify_albums[album]['energy'].append(features[0]['energy'])
        spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
        spotify_albums[album]['liveness'].append(features[0]['liveness'])
        spotify_albums[album]['loudness'].append(features[0]['loudness'])
        spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
        spotify_albums[album]['tempo'].append(features[0]['tempo'])
        spotify_albums[album]['valence'].append(features[0]['valence'])
        
        pop = sp.track(track)
        spotify_albums[album]['popularity'].append(pop['popularity'])
        track_count+=1


sleep_min = 2
sleep_max = 5
start_time = time.time()
request_count = 0

for i in spotify_albums:
    audio_features(i)
    request_count+=1
    if request_count % 5 == 0:
        print(str(request_count) + " playlists completed")
        time.sleep(np.random.uniform(sleep_min, sleep_max))
        #print('Loop #: {}'.format(request_count))
        #print('Elapsed Time: {} seconds'.format(time.time() - start_time))



dic_df = {}
dic_df['artist'] = []
dic_df['album'] = []
dic_df['track_number'] = []
dic_df['id'] = []
dic_df['name'] = []
dic_df['uri'] = []
dic_df['acousticness'] = []
dic_df['danceability'] = []
dic_df['energy'] = []
dic_df['instrumentalness'] = []
dic_df['liveness'] = []
dic_df['loudness'] = []
dic_df['speechiness'] = []
dic_df['tempo'] = []
dic_df['valence'] = []
dic_df['popularity'] = []

for album in spotify_albums: 
    for feature in spotify_albums[album]:
        dic_df[feature].extend(spotify_albums[album][feature])

df = pd.DataFrame.from_dict(dic_df)

cols = df.select_dtypes(include=[np.object]).columns
df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
df[cols] = df[cols].apply(lambda x: x.str.lower())

unique_artist = list(df['artist'].unique())
bands = pd.Series(bands).apply(lambda x: x.lower()).to_list()

def compare_lists(l1,l2):
  l2 = set(l2)
  unselected = [x for x in l1 if x not in l2]

  return unselected

df = df[~df.artist.isin(compare_lists(unique_artist,bands))]


df_spotify = df[['artist',
                 'album',
                 'name',
                 'acousticness',
                 'danceability',
                 'energy',
                 'instrumentalness',
                 'liveness',
                 'loudness',
                 'speechiness',
                 'tempo',
                 'valence',
                 'popularity']].sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()


X = df_spotify.copy(deep=True) 

categorical = X.select_dtypes(include = "object").columns
print(categorical)

LE = LabelEncoder()

for var in categorical:
    X[var] = LE.fit_transform(X[var].astype(str))


scaler = MinMaxScaler(copy=True, feature_range=(0,1))
X_scaled = scaler.fit_transform(X)


clust = range(1, 20)   
kmeans = [KMeans(n_clusters=i) for i in clust]

score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]


kmeans = Cluster()
#kmeans = kmeans.fit(X_scaled)
kmeans.fit(X_scaled)

print("Training complete!")

labels = kmeans.predict(X_scaled)

df_spotify['kmeans'] = labels

d={0:"workout",1:"concert",2:"instrumental"}
df_spotify['labels'] = df_spotify.kmeans.apply(lambda x:d[x])

Pkl_Filename = "./outputs/cluster_mood.pkl"  

with open(Pkl_Filename, 'wb') as file:  
  pickle.dump(kmeans, file)

print('Finished Training')