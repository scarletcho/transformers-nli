import json
import os

for root, dirs, files in os.walk("data"):
	for i, file in enumerate(files):
		files[i] = os.path.join(root, file)

claims = []
for i, file in enumerate(files):
	with open(file) as f:
		data = json.load(f)
		for item in data['theGraph']:
			if data['theGraph'][item]['type'] == 'Claim':
				claims.append(data['theGraph'][item]['text'])

pair_claims = []
for i in range(len(claims)):
	for j in range(i+1, len(claims)):
		pair_claims.append((claims[i], claims[j]))

# print(pair_claims)
print(len(pair_claims))

import pickle as pkl
with open("pair_claims.pkl", "wb") as f:
    pkl.dump(pair_claims,f)
