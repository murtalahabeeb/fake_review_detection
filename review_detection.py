import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import seaborn as sns
import time
import psutil
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
# Temporarily disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Initialize Sentiment Analyzer
nltk.download('vader_lexicon')

available_memory = psutil.virtual_memory().available
chunk_size = int(available_memory * 0.1 / (500 * 1024))  # Assume 500 bytes per row
chunk_size = max(chunk_size, 10000)  # Set a minimum chunk size of 10,000

# Read and process reviews in chunks
reviews_chunks = pd.read_json('/kaggle/input/business-reviews/review.json', lines=True, chunksize=chunk_size)
processed_reviews = []

for chunk in reviews_chunks:
    # Extract needed columns to reduce memory usage
    processed_chunk = chunk[['review_id', 'user_id', 'business_id', 'stars', 'text', 'useful', 'cool', 'funny']]
    processed_reviews.append(processed_chunk)


# Combine processed chunks
reviews_df = pd.concat(processed_reviews)




# Read and process users in chunks
users_chunks = pd.read_json('/kaggle/input/business-reviews/user.json', lines=True, chunksize=chunk_size)
processed_users = []

for chunk in users_chunks:
    # Extract needed columns to reduce memory usage
    processed_chunk = chunk[['user_id', 'review_count', 'useful', 'cool', 'funny', 'yelping_since']]
    processed_users.append(processed_chunk)

# Combine processed chunks
users_df = pd.concat(processed_users)



business = pd.read_json('/kaggle/input/business-reviews/business.json', lines=True)


# Merge dataframes
merged = reviews_df.merge(business, on='business_id', how='left')
merged = merged.merge(users_df, on='user_id', how='left')



sia = SentimentIntensityAnalyzer()

# Feature Engineering
enriched_and_labeled = []
# Define thresholds
low_sentiment_threshold = -0.8
high_sentiment_threshold = 0.8
short_review_length = 20
useful_ratio_threshold = 0.5

for df in merged:
    df['cool_x'] = df['cool_x'].fillna(0)
    df['funny_x'] = df['funny_x'].fillna(0)
    df['useful_x'] = df['useful_x'].fillna(0)
    # Drop rows where 'text' is NaN
    df.dropna(subset=['text'], inplace=True)

    # Ensure 'text' is a string
    df['text'] = df['text'].fillna("").astype(str)

    # Preprocess 'text'
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df['review_length'] = df['text'].apply(lambda x: len(x.split()))
    df['sentiment_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['is_short_review'] = df['review_length'] < 20
    df['useful_ratio'] = (df['useful_x'] + 1) / (df['cool_x'] + df['funny_x'] + 1)

    # Metadata Features
    df['high_activity_user'] = df['review_count_y'] > 50
    df['low_useful_user'] = df['useful_y'] < 10

    # Heuristics for Labeling
    
    df['review_length'] = pd.to_numeric(df['review_length'], errors='coerce')
    
    # Drop rows with NaN values in 'review_length'
    df = df.dropna(subset=['review_length'])
    
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    
    # Drop rows with NaN values in 'review_length'
    df = df.dropna(subset=['sentiment_score'])
    
    # Enhanced rule to decide fake reviews
    df['is_fake'] = (
        ((df['stars_x'] == 5) | (df['stars_x'] == 1)) & 
        (df['review_length'] < short_review_length)
    ) | (
        (df['high_activity_user']) & 
        (df['low_useful_user']) & 
        (df['useful_ratio'] < useful_ratio_threshold)
    ) | (
        (df['sentiment_score'] < low_sentiment_threshold) | 
        (df['sentiment_score'] > high_sentiment_threshold)
    )
        # Drop rows where 'text' is missing
    df = df.dropna(subset=['text'])

    # Optionally fill other missing values
    df['address'] = df['address'].fillna("Unknown")
    df['attributes'] = df['attributes'].fillna("No Info")
    df['hours'] = df['hours'].fillna("Not Provided")
    
    enriched_and_labeled.append(df)

df = pd.concat(enriched_and_labeled, ignore_index=True)
df = df.dropna(subset=['is_fake'])
# Save labeled dataset

print(f"Fake reviews detected: {df['is_fake'].sum()}")

# Separate the fake and genuine reviews
fake_reviews = df[df['is_fake'] == 1]
genuine_reviews = df[df['is_fake'] == 0]

#Ensure there are enough samples for equal distribution
num_samples = min(len(fake_reviews), len(genuine_reviews), 10000)  # Adjust as necessary

#Randomly sample 9000 from each class
sampled_fake_reviews = fake_reviews.sample(n=num_samples, random_state=42)
sampled_genuine_reviews = genuine_reviews.sample(n=num_samples, random_state=42)

#Combine the samples
df = pd.concat([sampled_fake_reviews, sampled_genuine_reviews]).sample(frac=1, random_state=42)  # Shuffle the dataset
df.to_csv('/kaggle/working/final_dataset.csv', index=False)

print(f"Duplicates in dataset: {df.duplicated().sum()}")
# Select features and target
features = ['stars_x', 'review_length','useful_ratio']
X = df[features]
y = df['is_fake']



label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
overlap = set(X_train.index).intersection(set(X_test.index))
print(f"Overlap between training and testing sets: {len(overlap)}")

y_train_series = pd.Series(y_train)


y_test_series = pd.Series(y_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Train set
y_train_series.value_counts().plot(kind='bar', color=['skyblue', 'orange'], ax=axes[0])
axes[0].set_title("Class Distribution in Training Dataset")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Genuine (0)", "Fake (1)"])

# Test set
y_test_series.value_counts().plot(kind='bar', color=['skyblue', 'orange'], ax=axes[1])
axes[1].set_title("Class Distribution in Testing Dataset")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Genuine (0)", "Fake (1)"])

# Save and display the plot
plt.tight_layout()
plt.savefig('class_distribution_side_by_side.png')
plt.show()

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Store results
model_results = {}
roc_data = {}

def plot_confusion_matrix(cm, classes, model_name, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.savefig(save_path)
    plt.show()

def plot_classification_metrics(report, model_name, save_path):
    # Convert classification report to a dictionary
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    scores = {metric: [report[cls][metric] for cls in classes] for metric in metrics}

    # Bar chart
    bar_width = 0.25
    index = np.arange(len(classes))

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, scores[metric], width=bar_width, label=metric.capitalize())

    plt.title(f"Precision, Recall, and F1-Score for {model_name}")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.xticks(index + bar_width, classes)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
# Function to train and evaluate a model with optional GridSearchCV
def train_and_evaluate_model_with_gridsearch(model, param_grid, model_name):
    # Perform GridSearchCV if param_grid is provided
    if param_grid:
        print(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

    start_time = time.time()
    model.fit(X_train, y_train)
  
    training_time = time.time() - start_time
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluation Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    model_results[model_name] = {"report": report, "confusion_matrix": conf_matrix, "training_time": training_time}

    # Print Metrics
    print(f"\n{model_name} - Classification Report:\n")
    print(report)
    
    print(f"\n{model_name} - Confusion Matrix:\n")
    print(conf_matrix)
    
    print(f"{model_name} - Training Time: {training_time:.2f} seconds")

  # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, classes=["Genuine", "Fake"], model_name=model_name, save_path=f'/kaggle/working/{model_name.lower().replace(" ", "_")}_conf_matrix.png')

    # Plot Precision, Recall, and F1-Score
    plot_classification_metrics(report, model_name, save_path=f'/kaggle/working/{model_name.lower().replace(" ", "_")}_metrics.png')

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f'/kaggle/working/roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    
  
    print(f"{model_name} Evaluation Complete!")
    return model

logistic_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

random_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

xgboost_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}
# Train and evaluate models with GridSearchCV
logistic_regression = train_and_evaluate_model_with_gridsearch(LogisticRegression(max_iter=500), logistic_params, "Logistic Regression")
random_forest = train_and_evaluate_model_with_gridsearch(RandomForestClassifier(random_state=42), random_forest_params, "Random Forest")
svm = train_and_evaluate_model_with_gridsearch(SVC(probability=True, random_state=42), svm_params, "SVM")
xgboost = train_and_evaluate_model_with_gridsearch(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgboost_params, "XGBoost")


# Generate Visualizations
plt.figure(figsize=(10, 8))

# ROC Curves
for model_name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=f"{model_name} (AUC = {data['auc']:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('/kaggle/working/roc_curves.png')
plt.show()

# Feature Importance (for Random Forest and XGBoost)
for model_name, model in zip(["Random Forest", "XGBoost"], [random_forest, xgboost]):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 6))
        plt.barh(features, model.feature_importances_)
        plt.title(f"Feature Importance - {model_name}")
        plt.xlabel("Importance Score")
        plt.savefig(f'/kaggle/working/{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.show()
