import pandas as pd

def get_categories(file_path):
  df = pd.read_csv(file_path)
  categories = df['category'].value_counts()
  return [(i, v) for i, v in categories.items()]


if __name__ == "__main__":
  file_name = './clean/l2.csv'
  print("Analysing file: {}".format(file_name))
  all_categories = get_categories(file_name)
  total_entries = sum([v for _, v in all_categories])
  print("There are {:,} entries in total".format(total_entries))
  for cat, val in all_categories:
    print("Category {} has {:,} entries. {:0.2f}%".format(cat, val, (val / total_entries) * 100))
    

