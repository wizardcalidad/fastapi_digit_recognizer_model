![ocr](https://geekgirljoy.files.wordpress.com/2017/05/ocr3.png)

This is a model that helps predict correctly the number on a hand-written digit image.

## THE FOLDER STRUCTURE
```
NEW
|
|- MACHINE-LEARNING-DIGIT-RECOGNITION
  |-DIGITS
  | |-APP.PY
  | |-DIGIT_RECOGNIZER.H5
  | |-DOCKERFILE
  | |-PROCFILE
  | |-REQUIREMENTS.TXT
  | |-RUNTIME.TXT
  | |-UTIL_DIGITS.PY
  |-.GITIGNORE
  |-DIGIT_RECOGNIZER_MODEL.IPYNB
  |-DIGITRECOGNITION.IPYNB
  |-LICENSE
  |-README.MD
  |-REQUIREMENTS.TXT

  ```

## THE GITHUB BRANCHES

There are three branches in total, namely:
- Indev
- Dev
- Main

## THE FASTAPI APPLICATION

The application was built on fastAPI with two major files, [app.py](https://github.com/wizardcalidad/machine-learning-digit-recognition/blob/indev/digits/app.py) and [util_digits.py](https://github.com/wizardcalidad/machine-learning-digit-recognition/blob/indev/digits/util_digits.py)

## REQUIREMENTS
There are two requirements.txt files, one inside our digits app and the other in the general folder. Everything inside our digits directory belongs to the app, including the requirements.txt file which is the requirement that the [*dockerfile*](https://github.com/wizardcalidad/machine-learning-digit-recognition/blob/indev/digits/Dockerfile) will adopt. 

```
fastapi==0.63.0
numpy==1.16.4
scikit-learn==0.20.1
uvicorn==0.9.0
tensorflow==2.5.0
uvloop==0.14.0
Keras==2.4.3
httplib2==0.14.0
httptools==0.1.1
gunicorn==20.0.4
python-multipart==0.0.5
```

## DOCKERFILE

```
FROM python:3.7-slim-buster
LABEL maintainer="Qoyum Yusuf --follow me on twitter @wizardcalidad01"
```
The dockerfile inherited from base image python:3.7-slim-buster because of irs lightness, it installed all requirements necessary, those in the digits app [requirements.txt](https://github.com/wizardcalidad/machine-learning-digit-recognition/blob/indev/digits/requirements.txt) file and expose port 5000.

## NOTEBOOK

There are two notebooks in the project folder, the two are different implementations of digit recognition models, but the one used for the purpose of this project is this: [digit recognition notebook](https://github.com/wizardcalidad/machine-learning-digit-recognition/blob/indev/digit_recognizer_model.ipynb).

## PROCFILE

This file is what helps our project build on Heroku cloud, the major requirement needs is gunicorn installed for cloud instead of uvicorn for the local running.

```
web: gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app
```

## RUNTIME.TXT

This is the python runtime environments required to run our application on cloud. and the one used is 

```
python-3.8.10
```

## THE APPLICATION
The link to the digit recognizer application is: 

Digit Application: [digit_app](https://digit-stage-app.herokuapp.com/)


## DOCUMENTATION

Read through the postman documentation on: [Postman](https://bit.ly/3cx6zpU) 


## MAINTAINER

The maintainer of this project work is Me, Myself and I. Follow me on: 

twitter: [The Wizard Calidad I of Mars](https://twitter.com/wizardcalidad01)

LinkedIn: [Qoyum Olatunde Yusuf](https://linkedin.com/in/wizardcalidad)

Hashnode: [The Wizardcalidad's AI Blog](wizardcalidad.hashnode.dev)# fastapi_digit_recognizer_model
