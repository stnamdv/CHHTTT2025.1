import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_training_data():
    """
    Create the training dataset based on the exercise table
    Returns: DataFrame with training data
    """
    # Define the training data from the exercise
    data = {
        'RID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
        'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
        'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
        'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
        'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    
    df = pd.DataFrame(data)
    return df

def manual_naive_bayes_calculation(df, new_example):
    """
    Perform manual Naive Bayes calculation step by step
    Args:
        df: Training dataframe
        new_example: Dictionary with new example features
    """
    print("=" * 60)
    print("MANUAL NAIVE BAYES CALCULATION")
    print("=" * 60)
    
    # Step 1: Calculate prior probabilities P(C)
    print("\n1. PRIOR PROBABILITIES P(C):")
    print("-" * 30)
    
    total_samples = len(df)
    class_counts = df['buys_computer'].value_counts()
    
    p_yes = class_counts['yes'] / total_samples
    p_no = class_counts['no'] / total_samples
    
    print(f"P(buys_computer = yes) = {class_counts['yes']}/{total_samples} = {p_yes:.4f}")
    print(f"P(buys_computer = no) = {class_counts['no']}/{total_samples} = {p_no:.4f}")
    
    # Step 2: Calculate conditional probabilities P(X|C)
    print("\n2. CONDITIONAL PROBABILITIES P(X|C):")
    print("-" * 40)
    
    # Filter data by class
    yes_data = df[df['buys_computer'] == 'yes']
    no_data = df[df['buys_computer'] == 'no']
    
    n_yes = len(yes_data)
    n_no = len(no_data)
    
    print(f"\nFor buys_computer = YES (n={n_yes}):")
    print("-" * 25)
    
    # Calculate P(X|C) for each feature
    features = ['age', 'income', 'student', 'credit_rating']
    conditional_probs = {}
    
    for feature in features:
        feature_value = new_example[feature]
        
        # Count occurrences in yes class
        count_yes = len(yes_data[yes_data[feature] == feature_value])
        p_x_given_yes = count_yes / n_yes
        
        # Count occurrences in no class
        count_no = len(no_data[no_data[feature] == feature_value])
        p_x_given_no = count_no / n_no
        
        print(f"P({feature}={feature_value}|yes) = {count_yes}/{n_yes} = {p_x_given_yes:.4f}")
        print(f"P({feature}={feature_value}|no) = {count_no}/{n_no} = {p_x_given_no:.4f}")
        
        conditional_probs[feature] = {
            'yes': p_x_given_yes,
            'no': p_x_given_no
        }
    
    # Step 3: Calculate posterior probabilities P(C|X)
    print("\n3. POSTERIOR PROBABILITIES P(C|X):")
    print("-" * 35)
    
    # Calculate P(X|yes) * P(yes)
    prob_yes = p_yes
    for feature in features:
        prob_yes *= conditional_probs[feature]['yes']
    
    # Calculate P(X|no) * P(no)
    prob_no = p_no
    for feature in features:
        prob_no *= conditional_probs[feature]['no']
    
    print(f"P(X|yes) * P(yes) = {prob_yes:.6f}")
    print(f"P(X|no) * P(no) = {prob_no:.6f}")
    
    # Normalize to get posterior probabilities
    total_prob = prob_yes + prob_no
    posterior_yes = prob_yes / total_prob
    posterior_no = prob_no / total_prob
    
    print(f"\nPosterior P(yes|X) = {posterior_yes:.4f}")
    print(f"Posterior P(no|X) = {posterior_no:.4f}")
    
    # Step 4: Make prediction
    print("\n4. PREDICTION:")
    print("-" * 15)
    prediction = 'yes' if posterior_yes > posterior_no else 'no'
    confidence = max(posterior_yes, posterior_no)
    
    print(f"Predicted class: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    return prediction, confidence

def sklearn_naive_bayes(df, new_example):
    """
    Use scikit-learn to perform Naive Bayes classification
    Args:
        df: Training dataframe
        new_example: Dictionary with new example features
    """
    print("\n" + "=" * 60)
    print("SCIKIT-LEARN NAIVE BAYES IMPLEMENTATION")
    print("=" * 60)
    
    # Prepare data for sklearn
    X = df[['age', 'income', 'student', 'credit_rating']]
    y = df['buys_computer']
    
    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()
    
    for column in X.columns:
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X[column])
        encoders[column] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Train Naive Bayes model
    nb_model = CategoricalNB()
    nb_model.fit(X_encoded, y_encoded)
    
    # Prepare new example for prediction
    new_example_encoded = []
    for feature in ['age', 'income', 'student', 'credit_rating']:
        value = new_example[feature]
        encoded_value = encoders[feature].transform([value])[0]
        new_example_encoded.append(encoded_value)
    
    # Make prediction
    prediction_encoded = nb_model.predict([new_example_encoded])[0]
    prediction = le_target.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probabilities
    probabilities = nb_model.predict_proba([new_example_encoded])[0]
    
    print(f"\nScikit-learn prediction: {prediction}")
    print(f"Probability for 'no': {probabilities[0]:.4f}")
    print(f"Probability for 'yes': {probabilities[1]:.4f}")

def main():
    """
    Main function to run the Naive Bayes exercise
    """
    print("NAIVE BAYES CLASSIFICATION EXERCISE")
    print("=" * 50)
    
    # Create training data
    df = create_training_data()
    
    # Display training data
    print("\nTRAINING DATA:")
    print("-" * 15)
    print(df.to_string(index=False))
    
    # Define the new example to classify
    new_example = {
        'age': '<=30',
        'income': 'medium',
        'student': 'yes',
        'credit_rating': 'fair'
    }
    
    print(f"\nNEW EXAMPLE TO CLASSIFY:")
    print("-" * 25)
    for feature, value in new_example.items():
        print(f"{feature}: {value}")
    
    # Perform manual calculation
    manual_prediction, manual_confidence = manual_naive_bayes_calculation(df, new_example)
    
    # Perform sklearn calculation
    sklearn_naive_bayes(df, new_example)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Manual calculation prediction: {manual_prediction}")
    print(f"Manual calculation confidence: {manual_confidence:.4f}")

if __name__ == "__main__":
    main() 