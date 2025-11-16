from datasets import load_dataset

dataset = load_dataset("2084Collective/prediction-markets-historical-v4")
def clean_dataset(x):
    accept = True
    accept = accept and x['platform'] == 'kalshi'
    accept = accept and (x['winning_outcome'].lower() == 'yes' or x['winning_outcome'].lower() == 'no')
    return accept
print("Prior to cleaning: ", len(dataset['train']))
cleaned_dataset = dataset.filter(clean_dataset)
print("After cleaning: ", len(cleaned_dataset['train']))

cleaned_dataset.push_to_hub("2084Collective/prediction-markets-historical-v4-cleaned")