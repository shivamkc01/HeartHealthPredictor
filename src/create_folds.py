import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import config

if __name__ == "__main__":
  df = pd.read_csv(config.SMOTE_FILE)
  df['skfold'] = -1

  df.sample(frac=1).reset_index(drop=False)

  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.SEED)
  y = df['class']

  for fold, (train_idx, valid_idx) in enumerate(skf.split(X=df, y=y)):
    df.loc[valid_idx, 'skfold'] = fold

  # saving the dataset with folds columns
  df.to_csv('../data/clean_data/df_10fold_smote.csv')