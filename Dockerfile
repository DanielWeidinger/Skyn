FROM tensorflow/serving
COPY ./models/saved_model /models/skyn
ENV MODEL_NAME skyn
