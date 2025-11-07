# Load and process the data
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Corrected CSV load
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Summer Project\diabetes_file.csv")
df.columns = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'class']

df1 = df[['glucose', 'bp', 'skin', 'insulin', 'bmi']]
df2 = df.drop(columns=['glucose', 'bp', 'skin', 'insulin', 'bmi'])

# Replace 0s with NaN
df1 = df1.replace(0, np.nan)

# Convert all columns to numeric to ensure mean calculation works
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column mean
df1.fillna(df1.mean(), inplace=True)

# Combine cleaned data
df3 = pd.concat([df1, df2], axis=1)


# Separate input and output
X = df3.drop(columns=['class'])
Y = df3['class']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# GUI
from tkinter import *
import tkinter.messagebox as m

w = Tk()
w.title("Diabetes Prediction System")

# Track trained models
trained_models = {}

# --- Model Functions ---
def lg():
    global L
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    try:
        L = LogisticRegression(max_iter=500)
        L.fit(X_train, Y_train)
        acc = round(accuracy_score(Y_test, L.predict(X_test)) * 100, 2)
        trained_models["LG"] = L
        selected_model.set("LG")
        highlight_selected()
        m.showinfo("Logistic Regression", f"Trained Successfully!\nAccuracy: {acc}%")
    except Exception as e:
        m.showerror("Error", str(e))

def knn():
    global K
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    try:
        K = KNeighborsClassifier(n_neighbors=5)
        K.fit(X_train, Y_train)
        acc = round(K.score(X_test, Y_test) * 100, 2)
        trained_models["KNN"] = K
        selected_model.set("KNN")
        highlight_selected()
        m.showinfo("KNN", f"Trained Successfully!\nAccuracy: {acc}%")
    except Exception as e:
        m.showerror("Error", str(e))

def dt():
    global D
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    try:
        D = DecisionTreeClassifier(random_state=7)
        D.fit(X_train, Y_train)
        acc = round(D.score(X_test, Y_test) * 100, 2)
        trained_models["DT"] = D
        selected_model.set("DT")
        highlight_selected()
        m.showinfo("Decision Tree", f"Trained Successfully!\nAccuracy: {acc}%")
    except Exception as e:
        m.showerror("Error", str(e))

def rf():
    global R
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    try:
        R = RandomForestClassifier(random_state=5)
        R.fit(X_train, Y_train)
        acc = round(R.score(X_test, Y_test) * 100, 2)
        trained_models["RF"] = R
        selected_model.set("RF")
        highlight_selected()
        m.showinfo("Random Forest", f"Trained Successfully!\nAccuracy: {acc}%")
    except Exception as e:
        m.showerror("Error", str(e))

# --- Compare Function ---
def compare():
    if not trained_models:
        m.showwarning("Warning", "Please train at least one model first.")
        return

    import matplotlib.pyplot as plt
    models = list(trained_models.keys())
    accuracies = []
    for name, model in trained_models.items():
        accuracies.append(round(model.score(X_test, Y_test) * 100, 2))

    plt.bar(models, accuracies, color=['orange', 'green', 'yellow', 'blue'][:len(models)])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.show()

# --- Submit Prediction ---
def submit():
    try:
        model_name = selected_model.get()
        if model_name not in trained_models:
            m.showwarning("Warning", "Please train and select a model first.")
            return

        entries = [Eglucose, Ebp, Eskin, Einsulin, Ebmi, Epreg, Epedigree, Eage]
        data = []
        for e in entries:
            val = e.get().strip()
            if not val.replace('.', '', 1).isdigit():
                raise ValueError("All inputs must be numeric.")
            data.append(float(val))

        data = np.array([data])
        result = trained_models[model_name].predict(data)[0]
        msg = "You have NO Diabetes" if result == 0 else "You HAVE Diabetes"
        m.showinfo("Prediction Result", msg)
    except Exception as e:
        m.showerror("Error", f"Please enter valid numeric values.\nDetails: {e}")

# --- Reset ---
def reset():
    for e in [Eglucose, Ebp, Eskin, Einsulin, Ebmi, Epreg, Epedigree, Eage]:
        e.delete(0, END)

# --- Highlight Selected Model ---
def highlight_selected():
    for btn_name, btn in button_dict.items():
        btn.config(bg='SystemButtonFace')
    sel = selected_model.get()
    if sel in button_dict:
        button_dict[sel].config(bg='lightgreen')

# --- GUI Layout ---
Label(w, text="Diabetes Prediction Using ML", font=('arial', 20, 'bold'), bg='pink').grid(row=1, column=1, columnspan=4)

selected_model = StringVar()
button_dict = {}

button_dict["LG"] = Button(w, text='LG', font=('arial', 15, 'bold'), command=lg)
button_dict["LG"].grid(row=2, column=1)
button_dict["KNN"] = Button(w, text='KNN', font=('arial', 15, 'bold'), command=knn)
button_dict["KNN"].grid(row=2, column=2)
button_dict["DT"] = Button(w, text='DT', font=('arial', 15, 'bold'), command=dt)
button_dict["DT"].grid(row=2, column=3)
button_dict["RF"] = Button(w, text='RF', font=('arial', 15, 'bold'), command=rf)
button_dict["RF"].grid(row=2, column=4)

Button(w, text="COMPARE", font=('arial', 15, 'bold'), command=compare).grid(row=3, column=2, columnspan=2)

Label(w, text="Predict for a New Person", font=('arial', 20, 'bold'), bg='pink').grid(row=4, column=1, columnspan=4)

def field(label, entry, row, col1, col2):
    label.grid(row=row, column=col1)
    entry.grid(row=row, column=col2)

Lglucose = Label(w, text="GLUCOSE", font=('arial', 15, 'bold'))
Eglucose = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
Lbp = Label(w, text="Blood Pressure", font=('arial', 15, 'bold'))
Ebp = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
field(Lglucose, Eglucose, 5, 1, 2)
field(Lbp, Ebp, 5, 3, 4)

Lskin = Label(w, text="Skin", font=('arial', 15, 'bold'))
Eskin = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
Linsulin = Label(w, text="Insulin", font=('arial', 15, 'bold'))
Einsulin = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
field(Lskin, Eskin, 6, 1, 2)
field(Linsulin, Einsulin, 6, 3, 4)

Lbmi = Label(w, text="BMI", font=('arial', 15, 'bold'))
Ebmi = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
Lpreg = Label(w, text="Preg", font=('arial', 15, 'bold'))
Epreg = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
field(Lbmi, Ebmi, 7, 1, 2)
field(Lpreg, Epreg, 7, 3, 4)

Lpedigree = Label(w, text="Pedigree", font=('arial', 15, 'bold'))
Epedigree = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
Lage = Label(w, text="Age", font=('arial', 15, 'bold'))
Eage = Entry(w, font=('arial', 15, 'bold'), width=10, bg='yellow')
field(Lpedigree, Epedigree, 8, 1, 2)
field(Lage, Eage, 8, 3, 4)

Button(w, text="SUBMIT", font=('arial', 15, 'bold'), command=submit).grid(row=9, column=1, columnspan=2)
Button(w, text="RESET", font=('arial', 15, 'bold'), command=reset).grid(row=9, column=3, columnspan=2)

w.mainloop()
