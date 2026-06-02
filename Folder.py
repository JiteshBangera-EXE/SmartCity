import os 

folders = [
    'data/raw', 'data/processsed', 'data/features',
    'notebooks', 'models', 'reports/figures', 'src',
]

for folder in folders:
    osmake = os.makedirs(folder, exist_ok=True)

print('Folders created successfully!')