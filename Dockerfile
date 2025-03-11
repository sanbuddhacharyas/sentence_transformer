From python:3.10

WORKDIR /usr/src/app

# Copy all the project files 
COPY . /usr/src/app

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx 

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000