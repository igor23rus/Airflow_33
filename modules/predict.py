from datetime import datetime
import json
import os
import dill
import pandas as pd



def predict():
    path = os.environ.get('PROJECT_PATH', '.')

    with open(f'{path}/data/models/cars_pipe_.pkl', 'rb') as file:
        model = dill.load(file)

    new_df = pd.DataFrame(columns=['ID', 'PRICE', 'RESULT'])

    for filename in os.listdir(f'{path}/data/test'):
        if filename.endswith('.json'):
            with open(os.path.join(f'{path}/data/test', filename)) as file:
                form = json.load(file)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)
                data = {'ID': df.id, 'PRICE': df.price, 'RESULT': y[0]}
                new_df = new_df._append(data, ignore_index=True)

    new_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
