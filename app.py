#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import flask


# In[2]:


from flask import Flask, request, jsonify, render_template
import pickle


# In[3]:


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


# In[4]:


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Cholesterol is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)


# In[ ]:





# In[ ]:




