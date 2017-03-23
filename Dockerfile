FROM python:2.7 

RUN pip install numpy scikit-learn pandas scipy

# add everything in the current local folder to the container: REQUIRED!
ADD . /

# run your submission by executing the entrypoint: REQUIRED!
ENTRYPOINT ["./run.sh"]

