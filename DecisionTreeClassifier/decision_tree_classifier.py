#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Classifier
Predict driver risk based on attributes
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Vietnamese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_training_data():
    """
    Create training data from the table in the exercise
    """
    print("=== CREATE TRAINING DATA ===")
    
    # Create training data from the table in the exercise
    training_data = {
        'ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'time': ['1-2', '2-7', '>7', '1-2', '>7', '1-2', '2-7', '2-7'],
        'gender': ['m', 'm', 'f', 'f', 'm', 'm', 'f', 'm'],
        'area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
        'risk': ['low', 'high', 'low', 'high', 'high', 'high', 'low', 'low']
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    print("Training data:")
    print(df)
    print()
    
    return df

def create_test_data():
    """
    Create test data (A, B, C) from the exercise
    """
    print("=== CREATE TEST DATA ===")
    
    # Create test data from the table in the exercise
    test_data = {
        'ID': ['A', 'B', 'C'],
        'time': ['1-2', '2-7', '1-2'],
        'gender': ['f', 'm', 'f'],
        'area': ['rural', 'urban', 'urban']
    }
    
    # Convert to DataFrame
    df_test = pd.DataFrame(test_data)
    print("Test data:")
    print(df_test)
    print()
    
    return df_test

def preprocess_data(df_train, df_test):
    """
    Preprocess data: encode categorical variables
    """
    print("=== PREPROCESS DATA ===")
    
    # Create LabelEncoder for each categorical attribute
    le_time = LabelEncoder()
    le_gender = LabelEncoder()
    le_area = LabelEncoder()
    le_risk = LabelEncoder()
    
    # Encode training data
    X_train_encoded = pd.DataFrame({
        'time': le_time.fit_transform(df_train['time']),
        'gender': le_gender.fit_transform(df_train['gender']),
        'area': le_area.fit_transform(df_train['area'])
    })
    
    y_train_encoded = le_risk.fit_transform(df_train['risk'])
    
    # Encode test data
    X_test_encoded = pd.DataFrame({
        'time': le_time.transform(df_test['time']),
        'gender': le_gender.transform(df_test['gender']),
        'area': le_area.transform(df_test['area'])
    })
    
    # Print mapping information
    print("Mapping for attribute 'time':")
    for i, label in enumerate(le_time.classes_):
        print(f"  {label} -> {i}")
    
    print("\nMapping for attribute 'gender':")
    for i, label in enumerate(le_gender.classes_):
        print(f"  {label} -> {i}")
    
    print("\nMapping for attribute 'area':")
    for i, label in enumerate(le_area.classes_):
        print(f"  {label} -> {i}")
    
    print("\nMapping for attribute 'risk':")
    for i, label in enumerate(le_risk.classes_):
        print(f"  {label} -> {i}")
    
    print()
    
    return X_train_encoded, y_train_encoded, X_test_encoded, le_time, le_gender, le_area, le_risk

def build_decision_tree(X_train, y_train):
    """
    Build decision tree from training data
    """
    print("=== BUILD DECISION TREE ===")
    
    # Create and train decision tree
    # criterion='entropy': use entropy to calculate information gain
    # random_state=42: for reproducible results
    tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    tree.fit(X_train, y_train)
    
    print("Decision tree trained successfully!")
    print(f"Depth of the tree: {tree.get_depth()}")
    print(f"Số lượng node: {tree.tree_.node_count}")
    print()
    
    return tree

def visualize_tree(tree, feature_names, class_names):
    """
    Visualize decision tree
    """
    print("=== VISUALIZE DECISION TREE ===")
    
    # Create figure with large size for better readability
    plt.figure(figsize=(20, 12))
    
    # Plot decision tree
    plot_tree(tree, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    
    plt.title('Decision Tree for Driver Risk Prediction', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Decision tree saved to 'decision_tree.png'")
    print()

def analyze_tree_structure(tree, feature_names):
    """
    Analyze the structure of the decision tree
    """
    print("=== ANALYZE DECISION TREE STRUCTURE ===")
    
    # Get information about nodes
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    
    print("Detailed structure of the decision tree:")
    print("-" * 50)
    
    for i in range(n_nodes):
        if children_left[i] == children_right[i]:  # Leaf node
            print(f"Node {i}: LEAF - Value: {value[i]}")
        else:  # Split node
            print(f"Node {i}: SPLIT - {feature_names[feature[i]]} <= {threshold[i]:.2f}")
            print(f"  -> Left: Node {children_left[i]}")
            print(f"  -> Right: Node {children_right[i]}")
        print()
    
    print("Explanation of the structure:")
    print("1. Root node uses the most important attribute to split")
    print("2. Child nodes continue to split based on other attributes")
    print("3. Leaf nodes contain the final prediction result")
    print()

def predict_test_data(tree, X_test, df_test, le_risk):
    """
    Predict results for test data
    """
    print("=== PREDICT RESULTS ===")
    
    # Perform prediction
    predictions_encoded = tree.predict(X_test)
    predictions = le_risk.inverse_transform(predictions_encoded)
    
    # Calculate prediction probabilities
    probabilities = tree.predict_proba(X_test)
    
    # Create result table
    results = df_test.copy()
    results['predicted_risk'] = predictions
    results['confidence_low'] = [f"{prob[0]:.3f}" for prob in probabilities]
    results['confidence_high'] = [f"{prob[1]:.3f}" for prob in probabilities]
    
    print("Kết quả dự đoán:")
    print(results.to_string(index=False))
    print()
    
    # Explain each prediction
    print("Detailed explanation:")
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"ID {row['ID']}:")
        print(f"  - Time: {row['time']}")
        print(f"  - Gender: {row['gender']}")
        print(f"  - Area: {row['area']}")
        print(f"  - Predicted risk: {row['predicted_risk']}")
        print(f"  - Confidence (low): {row['confidence_low']}")
        print(f"  - Confidence (high): {row['confidence_high']}")
        print()
    
    return results

def calculate_manual_entropy():
    """
    Calculate manual entropy to understand the process of building the tree
    """
    print("=== CALCULATE MANUAL ENTROPY ===")
    
    # Dữ liệu gốc
    data = {
        'time': ['1-2', '2-7', '>7', '1-2', '>7', '1-2', '2-7', '2-7'],
        'gender': ['m', 'm', 'f', 'f', 'm', 'm', 'f', 'm'],
        'area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
        'risk': ['low', 'high', 'low', 'high', 'high', 'high', 'low', 'low']
    }
    
    # Calculate entropy of the original data
    risk_counts = pd.Series(data['risk']).value_counts()
    total_samples = len(data['risk'])
    
    print("Distribution of risk in the original data:")
    for risk, count in risk_counts.items():
        p = count / total_samples
        print(f"  {risk}: {count}/{total_samples} = {p:.3f}")
    
    # Calculate entropy
    entropy = 0
    for count in risk_counts:
        p = count / total_samples
        if p > 0:
            entropy -= p * np.log2(p)
    
    print(f"Entropy of the original data: {entropy:.3f}")
    print()
    
    # Calculate Information Gain for each attribute
    print("Calculate Information Gain for each attribute:")
    
    for attribute in ['time', 'gender', 'area']:
        print(f"\nAttribute: {attribute}")
        
        # Group data by attribute
        df_temp = pd.DataFrame(data)
        grouped = df_temp.groupby(attribute)['risk'].value_counts().unstack(fill_value=0)
        
        print("Distribution:")
        print(grouped)
        
        # Calculate entropy for each value of the attribute
        weighted_entropy = 0
        for value in df_temp[attribute].unique():
            subset = df_temp[df_temp[attribute] == value]
            subset_risk_counts = subset['risk'].value_counts()
            subset_total = len(subset)
            
            subset_entropy = 0
            for count in subset_risk_counts:
                p = count / subset_total
                if p > 0:
                    subset_entropy -= p * np.log2(p)
            
            weight = subset_total / total_samples
            weighted_entropy += weight * subset_entropy
            
            print(f"  {value}: entropy = {subset_entropy:.3f}, weight = {weight:.3f}")
        
        information_gain = entropy - weighted_entropy
        print(f"Information Gain = {entropy:.3f} - {weighted_entropy:.3f} = {information_gain:.3f}")
    
    print()

def main():
    """
    Main function to run the entire process
    """
    print("=" * 60)
    print("DECISION TREE CLASSIFIER")
    print("Predict driver risk based on attributes")
    print("=" * 60)
    print()
    
    # Step 1: Create training data
    df_train = create_training_data()
    
    # Step 2: Create test data
    df_test = create_test_data()
    
    # Step 3: Preprocess data
    X_train, y_train, X_test, le_time, le_gender, le_area, le_risk = preprocess_data(df_train, df_test)
    
    # Step 4: Build decision tree
    tree = build_decision_tree(X_train, y_train)
    
    # Step 5: Visualize tree
    feature_names = ['time', 'gender', 'area']
    class_names = le_risk.classes_
    visualize_tree(tree, feature_names, class_names)
    
    # Step 6: Analyze tree structure
    analyze_tree_structure(tree, feature_names)
    
    # Step 7: Predict results
    # Dự đoán kết quả cho dữ liệu kiểm tra
    results = predict_test_data(tree, X_test, df_test, le_risk)
    
    # Step 8: Calculate manual entropy
    # Calculate manual entropy to understand the process of building the tree
    calculate_manual_entropy()
    
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main() 