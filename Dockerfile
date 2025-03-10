From python:3.10

WORKDIR /usr/src/app

# Copy all the project files 
COPY . /usr/src/app

RUN apt-get updateg
RUN apt-get install -y libgl1-mesa-lx 

RUN pip install --upgrage pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000