import cv2
import numpy as np
import tensorflow as tf
import collections

from models import facenet
from configs import config

with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():
        print('Loading feature extraction model')
        facenet.load_model(config.FACENET_MODEL_PATH)

        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        people_detected = set()
        person_detected = collections.Counter()

    def preprocess_face(face_img):
        scaled = cv2.resize(face_img, (config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE, 3)
        return scaled_reshape

    def compare_faces(faces, model, class_names):
        results = []
        for face in faces:
            scaled_face = preprocess_face(face)
            feed_dict = {images_placeholder: scaled_face, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            best_name = class_names[best_class_indices[0]]
            results.append((best_name, best_class_probabilities[0]))

        return results
