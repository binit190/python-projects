import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('admission_data.csv')

# Preprocess the data
X = df.drop(['admitted'], axis=1)  # features
y = df['admitted']  # target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Define a function to predict the chance of admission
def predict_chance_of_admission(gpa, gre, toefl, university_rating, sop, lor, cgpa):
    # Create a new data point
    new_data = pd.DataFrame({'gpa': [gpa], 'gre': [gre], 'toefl': [toefl], 'university_rating': [university_rating], 'sop': [sop], 'lor': [lor], 'cgpa': [cgpa]})
    
    # Make a prediction
    prediction = model.predict_proba(new_data)[:, 1]
    
    # Return the chance of admission
    return prediction[0]

# Example usage:
gpa = 3.5
gre = 320
toefl = 110
university_rating = 4
sop = 4
lor = 4
cgpa = 8.5

chance_of_admission = predict_chance_of_admission(gpa, gre, toefl, university_rating, sop, lor, cgpa)
print("Chance of Admission:", chance_of_admission)
