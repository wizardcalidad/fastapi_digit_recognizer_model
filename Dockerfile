FROM python:3.7-slim-buster
LABEL maintainer="Qoyum Yusuf --follow me on twitter @wizardcalidad01"

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /usr/src/digits
WORKDIR /usr/src/digits

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "digits.app:app"]