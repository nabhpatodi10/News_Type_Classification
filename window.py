from tkinter import *
import pandas as pd
import joblib

model=joblib.load("model.pkl")
vectorizer=joblib.load("Vectorizer.pkl")

root=Tk()
root.title("News Type Prediction")
root.geometry("500x500")

heading_var = StringVar()
desc_var = StringVar()

heading = Label(root, text="Heading:")
heading.place(relx=0.1, rely=0.05, relheight=0.05, relwidth=0.1)

heading_entry = Entry(root, textvariable=heading_var)
heading_entry.place(relx=0.22, rely=0.05, relwidth=0.2, relheight=0.05)

desc = Label(root, text="Description:")
desc.place(relx=0.1, rely=0.125, relheight=0.05, relwidth=0.125)

desc_entry = Entry(root, textvariable=desc_var)
desc_entry.place(relx=0.25, rely=0.125, relwidth=0.5, relheight=0.05)

result = Label(root)
result.place(relx=0.4, rely=0.4, relheight=0.05, relwidth=0.2)

def predict():
    if(heading_var.get()!="" or desc_var.get()!=""):
        custom_data = pd.DataFrame({
            'headline': [heading_var.get()],
            'short_description': [desc_var.get()]
        })

        custom_data['combined_text'] = custom_data['headline'] + ' ' + custom_data['short_description']
        test = vectorizer.transform(custom_data['combined_text'])
        result.configure(text=model.predict(test)[0])
    else:
        result.configure(text="Please Enter Data")
    root.update_idletasks()

pred_button = Button(root, text="Predict", command=predict)
pred_button.place(relx=0.45, rely=0.2, relwidth=0.1, relheight=0.05)

root.mainloop()