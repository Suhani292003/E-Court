#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 
# ## 1.1 Install dependencies

# In[202]:


get_ipython().system('pip install tensorflow==2.14.0 opencv-python matplotlib')


# ## 1.2 Import dependencies

# In[1]:


# Import standard dependencies 
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# Import tensorflow dependencies
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# ## 1.3 Set GPU Growth

# In[3]:


# Avoid OOM errors by setting GPU Memory Xonsumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    


# ## 1.4 Create Folder Structures 

# In[4]:


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# In[5]:


# Make the directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# ## 2. Collect Positives and Anchors
# 

# ## 2.1 Untar Labelled Faces in the Wild Dataset

# In[6]:


# http://vis-www.cs.umass.edu/Lfw/


# In[7]:


pwd


# In[8]:


cd d:


# In[9]:


cd Youtube


# In[10]:


pwd


# In[11]:


get_ipython().system('tar -xf lfw.tgz')


# In[12]:


# Move LFW Images to the following repository data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)


# ## 2.2 Collective Positive and Anchor Classes

# In[13]:


# Import uuid library to generate unique image names
import uuid


# In[14]:


os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))


# In[15]:


# Esstablish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking Gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()


# In[19]:


plt.imshow(frame[120:120+250,200:200+250, :])


# ## 3. Load and Preprocess Images

# ## 3.1 Get Image Directories

# In[20]:


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)


# In[21]:


dir_test = anchor.as_numpy_iterator()


# In[22]:


print(dir_test.next())


# ## 3.2 Preprocessing - Scale and Resize

# In[23]:


def preprocess(file_path):
    
    # Read in image from file path 
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1
    img = img / 255.0
    
    # Return image
    return img


# In[24]:


img = preprocess('data\\anchor\\7f4499fd-6c3c-11ee-ad1e-d6108804ce5f.jpg')


# In[25]:


img.numpy().max()


# ## 3.3 Create Labelled Dataset

# In[48]:


# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0


# In[49]:


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[50]:


samples = data.as_numpy_iterator() 


# In[51]:


exampple = samples.next()


# In[52]:


exampple


# ## 3.4 Build Train and Test Partition

# In[53]:


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# In[54]:


res = preprocess_twin(*exampple)


# In[55]:


plt.imshow(res[1])


# In[56]:


res[2]


# In[57]:


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


# In[58]:


samples = data.as_numpy_iterator()


# In[59]:


len(samples.next())


# In[64]:


samp = samples.next()


# In[65]:


plt.imshow(samp[0])


# In[66]:


plt.imshow(samp[1])


# In[67]:


samp[2]


# In[68]:


round(len(data)*.7)


# In[69]:


data


# In[70]:


# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# In[71]:


train_samples = train_data.as_numpy_iterator()


# In[72]:


train_sample = train_samples.next()


# In[73]:


len(train_sample[0])


# In[74]:


# Testing partition
test_data = data.skip(round(len(data)*.7))
print(test_data)
test_data = test_data.take(round(len(data)*.3))

test_data = test_data.batch(16)
print(test_data)
#test_data = test_data.batch(8)
print(test_data)


# ## 4. Model Engineering

# ### 4.1 Build Embedding Layer 

# In[75]:


inp = Input(shape=(105,105,3), name='input_image')


# In[76]:


c1 = Conv2D(64, (10,10), activation='relu')(inp)


# In[77]:


m1 = MaxPooling2D(64, (2,2), padding='same')(c1)


# In[78]:


c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)


# In[79]:


c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)


# In[80]:


c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


# In[81]:


mod = Model(inputs=[inp], outputs=[d1], name='embedding')


# In[82]:


mod.summary()


# In[83]:


def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


# In[84]:


embedding = make_embedding()


# In[85]:


embedding.summary()


# ### 4.2 Build Distance Layer 

# In[86]:


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# In[87]:


l1 = L1Dist()


# In[88]:


l1(anchor_embedding, validation_embedding)


# ### 4.3 Make Siamese Model

# In[89]:


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))


# In[90]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[91]:


siamese_layer = L1Dist()


# In[92]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[93]:


classifier = Dense(1, activation='sigmoid')(distances)


# In[94]:


classifier


# In[95]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='siameseNetwork')


# In[96]:


siamese_network.summary()


# In[97]:


def make_siamese_model():
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components 
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siameseNetwork')
    


# In[98]:


siamese_model = make_siamese_model()


# In[99]:


siamese_model.summary()


# ## 5. Training 

# ### 5.1 Setup Loss and Optimizer

# In[100]:


binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[101]:


opt = tf.keras.optimizers.Adam(1e-4) #0.0001


# ### 5.2 Establish Checkpoints

# In[102]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# ### 5.3 Build Train Step Function 

# In[103]:


test_batch = train_data.as_numpy_iterator()


# In[104]:


batch_1 = test_batch.next()


# In[105]:


X = batch_1[:2]


# In[106]:


np.array(X).shape


# In[107]:


@tf.function
def train_step(batch):
    
    # Record all of our operations
    with tf.GradientTape() as tape:
        
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y,yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss,siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss


# ### 5.4 Build Training Loop

# In[108]:


def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n EPOCH {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
    # Loop through each batch
    for idx, batch in enumerate(data):
        # Run train step here
        train_step(batch)
        progbar.update(idx+1)
        
    # Save checkpoints
    if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)


# ### 5.5 Train the model

# In[109]:


EPOCHS = 50


# In[110]:


train(train_data, EPOCHS)


# ## 6. Evaluate Model

# ## 6.1 Import metrics 

# In[111]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# ## 6.2 Make Predictions

# In[112]:


# Get a batch of test data 
test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[113]:


test_val


# In[114]:


test_var = test_data.as_numpy_iterator().next()


# In[115]:


test_data


# In[116]:


len(test_var[1])


# In[117]:


y_hat = siamese_model.predict([test_input, test_val])
y_hat


# In[118]:


# Post processing the results
[1 if prediction > 0.5 else 0 for prediction in y_hat]


# In[119]:


y_true


# ## 6.3 Calculate Metrics

# In[120]:


# Creating a metric object
m = Recall()

# calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[121]:


# Creating a metric object
m = Precision()

# calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy() 


# ## 6.4 Viz Results

# In[122]:


# Set plot size
plt.figure(figsize=(18,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[1])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[1])
plt.imshow()


# ## 7. Save Model

# In[123]:


# Save weights
siamese_model.save('siamesemodel.h5')


# In[124]:


# Reload model
model = tf.keras.models.load_model('siamesemodel.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[125]:


get_ipython().run_line_magic('pinfo2', 'tf.keras.models.load_model')


# In[126]:


# Make predictions with reloaded model
model.predict([test_input, test_val])


# In[128]:


# view model summary
model.summary()


# ## 8. Real Time Test

# ### 8.1 Verification Function

# In[ ]:


Application_data\verification_images


# In[130]:


os.path.join('Application_data', 'input_img', 'input_img.jpg')


# In[131]:


for image in os.listdir(os.path.join('Application_data', 'verification_images')):
    validation_img = preprocess(os.path.join('Application_data', 'verification_images', image))
    print(validation_img)
    


# In[1]:


def verify(model, detetction_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('Application_data', 'verification_images')):
        input_img = preprocess(os.path.join('Application_data', 'input_img', 'input_img.jpg'))
        validation_img = preprocess(os.path.join('Application_data', 'verification_images', image))
        
        # make predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('Application_data', 'verification_images')))
    verified = verification > verification_threshold
    
    return results, verified
    


# ### 8.2 OpenCV Real Time Verification

# In[2]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('Application_data', 'input_image', 'input_img.jpg'), frame)
        # Run verifications
        results, verified = verify(model, 0.5, 0.5)
        print(verified)
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()


# In[ ]:




