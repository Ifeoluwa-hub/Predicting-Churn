# create the base image
FROM python:3.8.12-slim

# install pipenv
RUN pip install pipenv

# specify the work directory
WORKDIR /app

# copy th pipfile to the current directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# run pipenv install without creating a virtual environment inside docker
RUN pipenv install --system --deploy

# copy predict.py and model file to the current directory
COPY ["predict.py", "model_C=1.0.bin", "./"]

# expose the port to the host machine
EXPOSE 9696

# specify the entry point
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"] 

#  && \
#     rm -rf /root/.cache

# Install the dependencies and packages in the requirements file
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt