import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv('data/loan_data.csv')

# Split Features and Target FIRST
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

# Identify Feature Types
num_feature = X.select_dtypes(include=['int64','float64']).columns
cat_feature = X.select_dtypes(include=['object']).columns

# Preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_feature),
    ('cat', cat_pipeline, cat_feature)
])

# Final Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LogisticRegression())
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, pred))

# Save Model 
import joblib
joblib.dump(model, 'model/loanData_model.pkl')