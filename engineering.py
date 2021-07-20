import pandas as pd


class FeatureEngineering:

    def __init__(self, df):
        self.df = df
        self.GetDummies()

    def GetDummies(self):
        self.df = pd.get_dummies(self.df,
                                 columns=["job", "education", "housing", "loan", "contact", "poutcome", "month", "marital"],
                                 prefix=["job", "edu", "house", "loan", "contact", "outcome", "month", "marital"])

