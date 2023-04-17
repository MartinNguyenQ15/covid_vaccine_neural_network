import pandas as pd

def clean_csv(file_name, out_name, to_drop):
  df = pd.read_csv(file_name)
  df = df.drop(columns=to_drop)
  df.to_csv(out_name, index=False, encoding='utf-8')

if __name__ == "__main__":
  clean_csv("./data/US_l2_vaccination_trending_searches.csv", "./clean/l2.csv", ["country_region", "country_region_code", "place_id", "query_type"])

