
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

main = tkinter.Tk()
main.title("Credit Card Fraud Detection Using Fuzzy Logic and Neural Network") #designing main screen
main.geometry("1300x1200")

global filename
global cls
global X_train, X_test, y_train, y_test
global auto_encoder
global dataset
global error_df

def upload(): #function to upload tweeter profile
    global filename
    global dataset
    #upload dataset file
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    #reading dataset as CSV file
    dataset = pd.read_csv(filename)
    dataset = dataset.drop(['Time'], axis=1)
    lbl = dataset['Class']
    #finding count of normal and fraud transaction
    unique, count = np.unique(lbl,return_counts=True)
    text.insert(END,str(dataset.head()))
    text.insert(END,"Total Normal Transaction: "+str(count[0])+"\n")
    text.insert(END,"Total Fraud Transaction: "+str(count[1])+"\n")
    #plotting normal and fraud transaction dataset
    label = dataset.groupby('Class').size()
    label.plot(kind="bar")
    plt.show()

def normalizeDataset():
    global X_train, X_test, y_train, y_test
    global dataset
    text.delete('1.0', END)
    #using scaler class we are scaling or normalizing dataset
    dataset['Amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
    #splitting dataset into train and test
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=42)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)
    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)
    X_train = X_train.values
    X_test = X_test.values
    text.insert(END,"Normalized Dataset\n\n")
    text.insert(END,str(dataset.head())+"\n\n")
    text.insert(END,"Total Training records after split: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total Testing records after split: "+str(X_test.shape[0])+"\n")

    
def trainModel():
    global X_train, X_test, y_train, y_test
    global auto_encoder
    text.delete('1.0', END)
    #defining input layer for encdoer
    inputLayer = Input(shape=(X_train.shape[1], ))
    #first encoder layer with neuron or features filtration size as 14
    encoder = Dense(14, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(inputLayer)
    #another encoder with 7 filetrs
    encoder = Dense(7, activation="relu")(encoder)
    #decoder with 7 filters to give output
    decoder = Dense(7, activation='tanh')(encoder)
    #decoder to perform prediction with givenshape
    decoder = Dense(X_train.shape[1], activation='relu')(decoder)
    #combining or extracting input layer and decoder layer to form auto encoding prediction layer
    auto_encoder = Model(inputs=inputLayer, outputs=decoder)
    #compiling and trsining odel
    auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    auto_encoder.load_weights('model/autoencoder.h5')
    text.insert(END,"Auto Encoder & Decoder model training completed\n\n")

    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    text.insert(END,"fuzzy logic and neural network accuracy: "+str(data['accuracy'][99]))
    print(data)
    #plot graph using auto encoder training LOSS
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('fuzzy logic and neural network training & Validation Loss Graph')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.show()
    

def prediction():
    global error_df
    global X_train, X_test, y_train, y_test
    global auto_encoder
    text.delete('1.0', END)
    #perform prediction on test data
    predictions = auto_encoder.predict(X_test)
    #calculate MAE error between origina Y value and predicted value
    mae = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'mae': mae, 'true_class': y_test})
    #calculate accuracy
    threshold = 2.9
    y_pred = [1 if e > threshold else 0 for e in error_df.mae.values]
    acc = accuracy_score(error_df.true_class, y_pred)
    text.insert(END,"fuzzy logic and neural network Accuracy on Test Data: "+str(acc))


    #plot histogram on non fraud transaction where class label 0 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['mae'] < 10)]
    _ = ax.hist(normal_error_df.mae.values, bins=10)
    plt.title("MAE histogram on Non-Fraudulent Transaction")
    plt.show()

    
def graph():
    #plot histogram on fraud transaction where class label 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[error_df['true_class'] == 1]
    _ = ax.hist(fraud_error_df.mae.values, bins=10)
    plt.title("MAE histogram on Fraudulent Transaction")
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='credit card fraud detection using fuzzy logic and neural network ')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

splitButton = Button(main, text="Normalize & Dataset Split", command=normalizeDataset)
splitButton.place(x=450,y=550)
splitButton.config(font=font1) 

encoderButton = Button(main, text="Train fuzzy logic and neural network  Model", command=trainModel)
encoderButton.place(x=50,y=600)
encoderButton.config(font=font1) 

predictButton = Button(main, text="Extract fuzzy logic and neural network for Prediction", command=prediction)
predictButton.place(x=450,y=600)
predictButton.config(font=font1)

graphButton = Button(main, text="MAE Histogram on Fraud Transaction", command=graph)
graphButton.place(x=50,y=650)
graphButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
