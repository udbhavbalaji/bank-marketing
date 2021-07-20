import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from engineering import FeatureEngineering


class ModelSelection(FeatureEngineering):

    def __init__(self, df):
        FeatureEngineering.__init__(self, df)
        label = LabelEncoder()
        x = self.df.drop(["default", "day_of_week", "y"], axis=1)
        y = self.df["y"]
        y_label = label.fit_transform(y)
        smote = SMOTE()
        x_sample, y_sample = smote.fit_resample(x, y_label)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_sample, y_sample, test_size=0.25)
        self.RandomForestClassifier()

    def RandomForestClassifier(self):
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(self.x_train, self.y_train)
        with open("model.pkl", "wb") as f:
            pickle.dump(forest, f)


def main():
    bank = pd.read_csv("bank-additional-full.csv", delimiter=";")
    ModelSelection(bank)
    return


if __name__ == "__main__":
    main()

