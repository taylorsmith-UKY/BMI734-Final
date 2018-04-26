import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical as cat
from sklearn.model_selection import StratifiedKFold

# ------------------------------- PARAMETERS ------------------------------------- #
data_path = 'data/w_zeroes/'
base_fname = 'ssx'
n_ex = 3815

n_epochs = 20
n_splits = 10
batch_size = 32
# -------------------------------------------------------------------------------- #

# Load inputs
data = []
for i in range(n_ex):
    data.append(np.loadtxt(data_path+base_fname+str(i)+'.txt'))
data = np.array(data)
# Add third dimension of 1 so Keras doesn't complain
data = np.expand_dims(data, -1)

# Load labels and make categorical (i.e. if 2 labels, each label is of form (0, 1) or (1, 0)
labels = np.loadtxt(data_path+'labels.txt', dtype=int)
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
    my_model.compile(loss='binary_crossentropy', optimizer='SGD')
    return my_model


# Stratify data into separate folds for cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)


for i, (train_idx, val_idx) in enumerate(skf.split(data, split_labels)):
    print('Evaluating on Fold ' + str(i+1) + ' of ' + str(n_splits) + '.')
    mcp_save = ModelCheckpoint(data_path+'CNN_Model_fold' + str(i+1) + '.h5',
                               save_best_only=True,monitor='val_loss', mode='min')

    X_train = data[train_idx]
    y_train = labels[train_idx]
    X_val = data[val_idx]
    y_val = labels[val_idx]

    model = get_model(data.shape[1:])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,
              callbacks=[mcp_save, ], validation_data=(X_val, y_val))
    loss = history.history['loss']
    loss = np.array(loss)
    lossFilename = data_path+'CNN_Model_fold' + str(i+1) + '_loss.txt'
    np.savetxt(lossFilename, loss, fmt='%.4f')
    print(model.evaluate(X_val, y_val))

