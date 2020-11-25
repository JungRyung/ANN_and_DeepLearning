#!/usr/bin/zsh
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


# In[4]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[5]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[6]:


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)


# In[ ]:





# In[ ]:





# In[ ]:




