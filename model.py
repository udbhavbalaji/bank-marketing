import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


class FeatureEngineering:

    def __init__(self, df):
        self.df = df
        self.GetDummies()

    def GetDummies(self):
        self.df = pd.get_dummies(self.df,
                                 columns=["job", "education", "housing", "loan", "contact", "poutcome", "month", "marital"],
                                 prefix=["job", "edu", "house", "loan", "contact", "outcome", "month", "marital"])


class ModelTraining(FeatureEngineering):

    def __init__(self, df):
        FeatureEngineering.__init__(self, df)
        label = LabelEncoder()
        x = self.df.drop(["default", "day_of_week", "y"], axis=1)
        y = self.df["y"]
        y_label = label.fit_transform(y)
        smote = SMOTE()
        x_sample, y_sample = smote.fit_resample(x, y_label)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_sample, y_sample, test_size=0.25)
        self.Table()

    def RandomForest(self):
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(self.x_train, self.y_train)
        prediction = forest.predict(self.x_test)
        precision = round(precision_score(self.y_test, prediction), 3)
        recall = round(recall_score(self.y_test, prediction), 3)
        f1 = round(f1_score(self.y_test, prediction), 3)
        return precision, recall, f1

    def GradientBoosting(self):
        boost = GradientBoostingClassifier()
        boost.fit(self.x_train, self.y_train)
        prediction = boost.predict(self.x_test)
        precision = round(precision_score(self.y_test, prediction), 3)
        recall = round(recall_score(self.y_test, prediction), 3)
        f1 = round(f1_score(self.y_test, prediction), 3)
        return precision, recall, f1

    def Table(self):
        precision_forest, recall_forest, f1_forest = self.RandomForest()
        precision_boost, recall_boost, f1_boost = self.GradientBoosting()
        table = pd.DataFrame({"Statistic Measure": ["Precision", "Recall", "F1_score"],
                              "Random Forest": [precision_forest, recall_forest, f1_forest],
                              "Gradient Boosting": [precision_boost, recall_boost, f1_boost]})
        return table


def main():
    bank = pd.read_csv("bank-additional-full.csv", delimiter=";")
    performance = ModelTraining(bank).Table()
    print(performance)
    return


if __name__ == "__main__":
    main()

