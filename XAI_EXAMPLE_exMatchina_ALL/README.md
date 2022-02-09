# ExMatchina
A Deep Neural Network explanation-by-example library for generating meaningful explanations. Used in our [Explainability Study](https://github.com/nesl/Explainability-Study)

## Prerequisites
Install the required Python packages
```Python
pip3 install -r requirements.txt
```

## Usage
1. Import the ExMatchina class
```Python
from ExMatchina import ExMatchina
```
2. Load ExMatchina with a particular TensorFlow model + example prototypes (e.g. training data)

```Python
# X_train.npy: a numpy array of prototypes
# model: the model of interest
training_data = np.load('./X_train.npy')
model = load_model('./model')

# selected_layer: the layer to use in identifying examples.
# We recommend the layer immediately following the last convolution (e.g. flatten layer)
selected_layer = "Flatten_1"

exm = ExMatchina(model=model, layer=selected_layer, examples=training_data)
```

3. Fetch examples and corresponding indices for a given input

```Python
# X_train.npy: a numpy array of model inputs
test_data = np.load('./X_test.npy')
test_input = test_data[0]
(examples, indices) = exm.return_nearest_examples(test_input)
```

## Examples
The `Examples/` folder contains the tutorial in python notebooks on using Exmatchina for different types of input data

### Data
Here's the Google Drive Link to the preprocessed data: [Link](https://drive.google.com/drive/folders/1ZRWIeUHxGbKpqWkJ2HpiSLtmUyllfThf?usp=sharing)

Download each of the folders there and place them in `Examples/data/`

### Trained Models
Inside the `trained_models/` folder, there are the pretrained models, named as `[domain].hdf5` for each of the domains: image, text, ECG

## BibTex

If you find this code and results useful in your research, please cite:

	@article{jeyakumar2020can,
	  title={How Can I Explain This to You? An Empirical Study of Deep Neural Network Explanation Methods},
	  author={Jeyakumar, Jeya Vikranth and Noor, Joseph and Cheng, Yu-Hsi and Garcia, Luis and Srivastava, Mani},
	  journal={Advances in Neural Information Processing Systems},
	  volume={33},
	  year={2020}
	}
