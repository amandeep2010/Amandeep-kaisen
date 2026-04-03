import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv('gesture_data.csv')
    except FileNotFoundError:
        print("Error: gesture_data.csv not found.")
        return

    # Extract labels and features
    if 'label' not in df.columns:
        print("Error: 'label' column not found in dataset.")
        return

    y = df['label']
    X = df.drop('label', axis=1)

    print("Splitting dataset into 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("MODEL ACCURACY METRICS")
    print("="*40)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*40)

    model_path = 'jjk_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel exported successfully to {model_path}.")

if __name__ == "__main__":
    main()
