# base image
FROM python:3.7-slim


# gunicorn's web server root
WORKDIR /home/site/wwwroot

# install required packages
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install eventlet==0.30.2 && \
    pip install -r requirements.txt

# copy all files
COPY . .

# allow to execute the wait-for-it script
# script source: https://github.com/vishnubob/wait-for-it
RUN chmod +x wait-for-it.sh

# expose port 80
EXPOSE 80

# start flask app (assumes file api.py with variable 'app' in it)
ENV FLASK_APP="src/api.py"
CMD flask run --host 0.0.0.0 --port 80
