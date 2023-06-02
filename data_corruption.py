import pandas as pd, numpy as np

def distance_calculation(coordinates):
    pass

def preprocessing(fname):
    data = pd.read_csv(fname)
    data = data.groupby('event_num')

    # data['position'] = data[['x', 'y', 'z']].values.tolist()
    # data = data.drop(['x', 'y', 'z'], axis=1)
    print(data.head())

def main():
    preprocessing('scrapped_data_mul1_small.csv')

if __name__ == "__main__":
    main()