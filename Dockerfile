FROM tensorflow/serving
COPY ./models/skyn/v1 /models/skyn 
ENV MODEL_NAME skyn
