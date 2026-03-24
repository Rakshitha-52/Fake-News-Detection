import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print(fake.head())
print(true.head())

fake['label'] = 0   # fake
true['label'] = 1   # real

data = pd.concat([fake, true])

print(data.head())
print(data.shape)

data = data[['text', 'label']]
print(data.head())

print(data['label'].value_counts())

data = data.sample(frac=1).reset_index(drop=True)

print(data.shape)