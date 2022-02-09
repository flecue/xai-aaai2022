
import numpy as np
from heapq import nlargest
from os import path
from tensorflow.keras.models import  Model


class ExMatchina:
    def package_single_input(self, input_data):
        return np.expand_dims(input_data,axis=0)

    def generate_labels(self, inputs, savefilepath=None):
        predictions = self.model.predict(inputs)
        labels = np.argmax(predictions,axis=1)
        if savefilepath:
            np.save(savefilepath, labels)
        return labels

    def get_label_for(self, input_data):
        inputs = self.package_single_input(input_data)
        return self.generate_labels(inputs)[0]


    def __init__(self, model, layer, examples, activations_file=None, labels_file=None):
        self.model = model
        self.layer = layer
        self.examples = examples


        # get activations
        print("Getting activations...")
        if activations_file and path.exists(activations_file):
            self.activations = np.load(activations_file)
        else:
            self.activations = get_activations(model, layer, examples, savefilepath=activations_file)

        # get labels
        print("Getting labels...")
        if labels_file and path.exists(labels_file):
            self.labels = np.load(labels_file)
        else:
            self.labels = self.generate_labels(examples, savefilepath=labels_file)
        


        # build activation matrix
        print("Generating activation matrix...")
        self.activation_matrix = np.zeros(shape=(len(self.activations[0]), len(self.activations)))
        for i in range(len(self.activations)):
            normalized = self.activations[i] / np.linalg.norm(self.activations[i])
            self.activation_matrix[:,i] = normalized


    def return_nearest_examples(self, test_input, num_examples=3):
        # test_input = np.expand_dims(test_input,axis=0)
        # label = np.argmax(self.model.predict(test_input))
        
        packaged_test = self.package_single_input(test_input)

        label = self.get_label_for(test_input)

        test_activation = get_activations(self.model, self.layer, packaged_test)[0]
        test_act_norm = test_activation / np.linalg.norm(test_activation)
        results = test_act_norm.dot(self.activation_matrix)
        topk = nlargest(num_examples, range(len(results)), key=lambda idx: results[idx] if label == self.labels[idx] else 0)
        return ([self.examples[idx] for idx in topk],topk)


def get_activations(model, layer_name, input_data, savefilepath=None):

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    
    intermediate_output = intermediate_layer_model.predict(input_data)
    if savefilepath:
        np.save(savefilepath, intermediate_output)
    return intermediate_output