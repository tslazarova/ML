from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 1.1 Import data and labels on which we want to train the model 
data = np.loadtxt('traindata.dat') # should include all N Signals ! 
data = data.astype('float32')
labels =  np.loadtxt('trainlabels.dat')
labels = labels.astype('float32')

# 1.2 Split data and labels on which we train our model into train and test sets
train_to_test_ratio = 0.66
data_train,data_test,labels_train,labels_test = train_test_split(data,labels,train_size=train_to_test_ratio)

# 2.1 Instantiate a Keras tensor
inputs = keras.Input(shape=(1024,),dtype='float32') 

# 2.2 Topology of the model
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(1024,)),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(256,activation='sigmoid'),
    keras.layers.Dense(5,activation='sigmoid')
    ])

# 2.3 Compile the model 
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # check loss, optimizer

# 2.4 Fit the model
history = model1.fit(data,labels,epochs=50, batch_size=128, validation_split=0.5, shuffle=True, verbose=1)

# 3.1 Evaluate the model - Calculate accuracy and validation_accuracy
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

'''
# 3.2 Create a plot of the accuracy of the model.fit 
epochs_range = range(1, len(acc) + 1)
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.plot(epochs_range, acc, label='Training accuracy')
plt.legend(loc='upper right')
title='Model2: N_Signals=2, Noise=0.2mV' # data, labels are taken from model.fit data, labels
plt.title(title)
#plt.savefig('Figures/model2_n5_noise05.png')
#plt.show()
'''
 # 4.1 Apply model to other datasets with different number of NSignals and NNoise

in_file=open("../all_data_labels.txt","r")
n=in_file.read().splitlines()
for i in range(len(n)-1):
    d,l= n[i].split()
    new_data=np.loadtxt(d)
    new_labels=np.loadtxt(l)    
    history = model1.fit(new_data,new_labels,epochs=50, batch_size=128, validation_split=0.5, shuffle=True, verbose=1)
    acc=history.history['accuracy']
    #print(type(acc[1]))
    val_acc=history.history['val_accuracy']
    epochs_range = range(1, len(acc) + 1) #?!
    title=str(d)
    new_acc=str(acc[1])
    plt.plot(epochs_range, val_acc, label='Validation accuracy')
    plt.plot(epochs_range, acc, label='Training accuracy')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()
    out_file = open('new_model_accuracy.txt', 'a')
    #out_file.writelines(f'{title} : Accuracy for train data: {acc*100:.2f} \n')
    out_file.writelines(title + ':' + new_acc + '\n')
#    break
