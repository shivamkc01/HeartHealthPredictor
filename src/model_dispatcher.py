import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import config 


df = pd.read_csv(config.SMOTE_10FOLD_FILE, na_values="?")
class_counts = df['class'].value_counts().sort_index()
class_priors = class_counts / class_counts.sum()


models = {
  'lr' : LogisticRegression(max_iter=100000, class_weight={0:1, 1:3,2:3, 3:6, 4:4}),
  'dt' : DecisionTreeClassifier(max_depth=5, min_samples_split=7,min_samples_leaf=4),
  'nb': GaussianNB(priors=class_priors.values),
  # 'rf': RandomForestClassifier()
}