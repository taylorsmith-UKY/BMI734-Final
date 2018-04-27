import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score,\
     precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

# ------------------------------- PARAMETERS ------------------------------------- #
# Data params
dataset = 'w_zeroes'    # Dataset selection
modelName = 'cnnA_250'      # Directory name for saved model information
base_fname = 'ssx'      # filenames ssx#.txt
n_ex = 3815

# Training params
n_epochs = 250          # Number of epochs to train each split
n_splits = 10            # Number of splits for cross-validation
batch_size = 32         # Training batch size
# -------------------------------------------------------------------------------- #

data_path = 'data/' + dataset + '/'
model_path = 'models/' + dataset + '/' + modelName + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Load inputs
data = []
for i in range(n_ex):
    data.append(np.loadtxt(data_path+base_fname+str(i)+'.txt'))

same_length = False
if np.all([len(data[x]) == len(data[x+1]) for x in range(n_ex - 1)]):
    same_length = True
    data = np.array(data)

    # Add third dimension so Keras doesn't complain
    data = np.expand_dims(data, -1)
    shp = data.shape[1:]


# Load labels and make categorical (i.e. if 2 labels, each label is of form (0, 1) or (1, 0)
labels = np.loadtxt(data_path+'labels.txt', dtype=int)[:n_ex]
split_labels = labels
labels = cat(labels, 2)

# Get Model
def get_model(in_shape):
    inputs = Input(in_shape, name='Image-Input')
    conv_1 = Conv2D(16,
                    kernel_size=(2, 2),
                    padding='same', activation='relu', name='Conv1')(inputs)
    pool_1 = MaxPooling2D((2, 2), name='Pool1')(conv_1)
    conv_2 = Conv2D(32,
                    kernel_size=(2, 2),
                    padding='same', activation='relu', name='Conv2')(pool_1)
    pool_2 = MaxPooling2D((2, 2), name='Pool2')(conv_2)
    flat = Flatten()(pool_2)
    x = Dense(2, activation='softmax', name='predictions')(flat)
    my_model = Model(input=inputs, output=x)
    my_model.compile(loss='binary_crossentropy', optimizer='SGD', )
    return my_model


# Stratify data into separate folds for cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

model = get_model(shp)
# Open the file
with open(model_path + 'model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

for i, (train_idx, val_idx) in enumerate(skf.split(data, split_labels)):
    print('Evaluating on Fold ' + str(i+1) + ' of ' + str(n_splits) + '.')

    # Checkpoint to save model params
    mcp_save = ModelCheckpoint(model_path+'trained_model_fold' + str(i+1) + '.h5',
                               save_best_only=True, monitor='val_loss', mode='min')

    # Get the training and test sets
    X_train = data[train_idx]
    y_train = labels[train_idx]
    X_val = data[val_idx]
    y_val = labels[val_idx]

    # Load and fit the model
    model = get_model(data.shape[1:])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,
                        callbacks=[mcp_save, ], validation_data=(X_val, y_val))
    loss = history.history['loss']
    loss = np.array(loss)

    # Plot the loss vs. number of epochs
    fig = plt.figure()
    plt.plot(np.arange(len(loss))+1, loss)
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch Number')
    plt.title('Validation Loss vs. Epoch\nFold ' + str(i+1))
    plt.savefig(model_path + 'fold' + str(i+1) + '_loss.png')
    plt.close(fig)

    # Plot the precision vs. recall curve
    y_pred = model.predict(X_val)
    pcurve, rcurve, _ = precision_recall_curve(y_val[:, 0], y_pred[:, 0])
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.step(rcurve, pcurve, color='b', alpha=0.2,
             where='post')
    plt.fill_between(rcurve, pcurve, where=None, alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AP=%0.2f' % average_precision_score(y_val[:, 0], y_pred[:, 0]))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val[:, 0], y_pred[:, 0])
    roc_auc = auc(fpr, tpr)
    plt.subplot(122)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(model_path + 'fold' + str(i + 1) + '_evaluation.png')


