import pandas as pd

# report header info
location = 'Seward'
routes = ['East', 'West']
stations = [1, 2, 3, 4, 5, 6, 7, 8]
type = 'Stationary'
date = None
start_time = None
duration = 5
party_size = 1
complete = True
temp = None
wind = None
precip = None
peds = None
dogs = None
offleash = None

# species codes / names
band_codes = pd.read_csv('/Users/jonathanwoodard/bcs_nbp/IBP-AOS-LIST24.csv')
cols = [c.lower() for c in band_codes.columns]
band_codes.columns = cols
cols = ['spec', 'commonname', 'sciname', 'spec6']
band_codes = band_codes[cols]

taxonomy = pd.read_csv('/Users/jonathanwoodard/bcs_nbp/ebird-taxonomy.csv')
band_taxonomy = band_codes.merge(taxonomy, left_on = 'sciname', right_on='scientific_name')

data_head = pd.read_csv('/Users/jonathanwoodard/bcs_nbp/spw_20230812.csv', nrows=1)
data_cols = ['station', 'start time', 'code', 'seen', 'heard', 'flyover', 'notes']
data = pd.read_csv('/Users/jonathanwoodard/bcs_nbp/spw_20230812.csv', header=3, usecols=data_cols)
data[data_cols[:2]] = data[data_cols[:2]].fillna(method='ffill')
data['station'] = data['station'].astype(int)
data[data_cols[3:6]] = data[data_cols[3:6]].fillna(0).astype(int)
data['code'] = data.code.apply(lambda x: x.upper())
data['location_name'] = f"{data_head.loc[0, 'location']} {data_head.loc[0, 'loop']} St" + data['station'].astype(str)

# merge data file with taxonomy on 4 letter band code
data_taxonomy = data.merge(band_taxonomy, left_on='code', right_on='spec', how='left')
# identify rows which didn't match
data_taxonomy[data_taxonomy.spec.isna()]
