import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_sample_data():
    data = {
        "cgpa": [7.5, 8.2, 6.8, 9.0, 7.1, 8.7, 6.5, 8.0, 7.8, 9.1],
        "internships": [1, 2, 0, 3, 1, 2, 0, 2, 1, 3],
        "projects": [2, 3, 1, 4, 2, 3, 1, 3, 2, 4],
        "aptitude_score": [65, 78, 55, 88, 62, 82, 50, 75, 68, 91],
        "placement": [
            "Not Placed",
            "Placed",
            "Not Placed",
            "Placed",
            "Not Placed",
            "Placed",
            "Not Placed",
            "Placed",
            "Placed",
            "Placed",
        ],
    }

    return pd.DataFrame(data)


def train_model():
    df = create_sample_data()

    X = df.drop("placement", axis=1)
    y = df["placement"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Student Placement Prediction")
    print("-----------------------------")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2f}")

    return model, accuracy


if __name__ == "__main__":
    train_model()
