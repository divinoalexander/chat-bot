# Chatbot Deployment with Flask and JavaScript

Deployment:
- Deploy within Flask app with jinja2 template

## Initial Setup:
This repo currently contains the starter files.

create a virtual environment
```
$ python3 -m venv venv
$ . venv/bin/activate
```
Install dependencies
```
$ pip install Flask torch torchvision nltk
```
Install nltk package
```
$ python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` with different intents and responses for your Chatbot

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python chat.py
```

implement `app.py` and `app.js`.
