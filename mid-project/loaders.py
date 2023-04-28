import pandas as pd

df = pd.read_csv("./cali_dataset.csv")

def __strategy_administered_date(labels):
    reduction = df['administered_date'].str.contains(r'|'.join(labels), regex=True)
    def matcher(value):
        return df['administered_date'].str.contains(value), 'administered_date'
    conversion = matcher
    return reduction, conversion

def __strategy_county(labels):
    reduction = df['county'].isin(labels)
    def matcher(value):
        return df['county'] == value, 'county'
    conversion = matcher
    return reduction, conversion


def load_by_year(labels_list=['2021', '2022', '2023'], selected_columns=['administered_date', 'pfizer_doses', 'moderna_doses', 'jj_doses']):
    return __load(__strategy_administered_date, labels_list, selected_columns)

# Issue to resolve with f1_score if more than 2 cities
def load_by_county(county_list = ['Calaveras','Plumas'], selected_columns = ['county', 'pfizer_doses', 'moderna_doses', 'jj_doses']):
    return __load(__strategy_county, county_list, selected_columns)


def __load(strategy, labels_list, selected_columns):
    reduction, conversion = strategy(labels_list)
    without = df.loc[reduction]
    for i, label in enumerate(labels_list):
        without.loc[conversion(label)] = i
    fval = 1 if len(selected_columns) == 1 else len(selected_columns) - 1
    return without[selected_columns], (-fval, fval)

