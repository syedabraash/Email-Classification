# Install required libraries
!pip install pandas scikit-learn

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset into a DataFrame
df = pd.read_csv('/content/email.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Display the distribution of labels
print(df['Category'].value_counts())

# Example of text cleaning (simple)
df['text'] = df['Message'].str.lower().str.replace('[^\w\s]', '')

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Labels
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_emails = ["Congratulations! You've won a $1,000 gift card.",
              "Can we schedule a meeting for tomorrow?"]
new_emails_transformed = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_transformed)
print(predictions)

# Get user input and classify emails
custom_email = input("Enter Email Message : ")
email_transformed = vectorizer.transform([custom_email])
prediction = model.predict(email_transformed)
print(prediction)
