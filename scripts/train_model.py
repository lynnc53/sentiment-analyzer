import numpy as np 
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

# load features and labels 
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# split the data 
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# logfistic regression model
log_model = LogisticRegression(C=1.0,max_iter=1000)
log_model.fit(X_train, y_train)

# multinomial naive bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# evaluate models 
def evaluate_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    print(f"\n---{title}---")
    print(classification_report(y_test,y_pred,target_names=["Negative","Positive"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# calling function 
evaluate_model(log_model, X_test, y_test, "Logistic Regression")
evaluate_model(nb_model, X_test, y_test, "Multinomial Naive Bayes")

# saving best model 
joblib.dump(log_model, "models/logistic_model.pkl")
