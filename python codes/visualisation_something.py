import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def main():
    # Load your data into a DataFrame, assuming it's named 'data'
    data = pd.read_csv('ai.csv')  # Replace 'your_data.csv' with the actual file path
    
    # Split the data into feature data (X) and target data (Y)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # Split the data into 75% training data and 25% testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

    # Feature scaling: Scale values in data between 0 and 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Use Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
    forest.fit(X_train, Y_train)

    # Test model accuracy on training data
    training_accuracy = forest.score(X_train, Y_train)

    # Test model accuracy on testing data
    testing_accuracy = forest.score(X_test, Y_test)

    # Generate confusion matrix
    cm = confusion_matrix(Y_test, forest.predict(X_test))

    # Extract variables from confusion matrix
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    # Print confusion matrix
    print(cm)

    # Print model accuracy on test data
    print('Model test accuracy =', (TP + TN) / (TP + TN + FP + FN))

if __name__ == "__main__":
    main()
