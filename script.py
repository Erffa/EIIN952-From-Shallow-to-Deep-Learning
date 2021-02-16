# IMPORTS
# file system exploration / management
import os, sys
# load the dataset
import torchvision
# image processing
import cv2
import numpy as np
from PIL import Image
# to read/parse xml files
from collections import OrderedDict
import json
import xmltodict
# nice displays
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
# colab elements
from google.colab import drive, output
from IPython.display import HTML, Javascript
# neural network kinda stuff
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential

# Information in text files

def get_all_indices_mode(path, mode):
    """
    """
    indices = []
    with open(path + f"{mode}.txt", 'r') as filehandle:
        for line in filehandle:
            idx = line.rstrip()
            indices.append(idx)
    return np.array(indices)

def get_all_indices_category(path, category, mode):
    """
    mode : either "train", "trainval" or "val"
    """
    filenames = []
    values = []
    with open(path + f"{category}_{mode}.txt", 'r') as filehandle:
        for line in filehandle:
            idx, val = line.rstrip().split(' ', 1)
            filenames.append(idx)
            values.append(int(val))
    return np.array(filenames), np.array(values)


# Categories selection
CATEGORIES = []

def update_categories(*args):
    global CATEGORIES
    CATEGORIES = args

output.register_callback('update_categories', update_categories)

CATEGORIES_HTML = HTML("""
<!DOCTYPE html>
<html>

<header>
<title>My form</title>
<meta charset="UTF-8"> 
<style>
form {
	text-align: center;
	width: 500px;
}
.family-col{
	display: inline-block;
	vertical-align: top;
	text-align: left;
	width: 120px;
}
</style>
</header>

<body>
<h2> Choose your categories </h2>
<form>
	<div id="col-handler">

		<div id="person" class="family-col">
			<input class="category" type="checkbox" id="male" name="gender" value="person"> <label for="person">Person</label><br>
		</div>

		<div id="animal" class="family-col">
			<input class="category" type="checkbox" id="male" name="gender" value="bird"> <label for="bird">Bird</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="cat"> <label for="cat">Cat</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="cow"> <label for="cow">Cow</label><br>
			<input class="category" type="checkbox" id="male" name="gender" value="dog"> <label for="dog">Dog</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="horse"> <label for="horse">Horse</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="sheep"> <label for="sheep">Sheep</label>
		</div>

		<div id="vehicule" class="family-col">
			<input class="category" type="checkbox" id="male" name="gender" value="aeroplane"> <label for="aeroplane">Aeroplane</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="bicycle"> <label for="bicycle">Bicycle</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="boat"> <label for="boat">Boat</label><br>
			<input class="category" type="checkbox" id="male" name="gender" value="bus"> <label for="bus">Bus</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="car"> <label for="car">Car</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="motorbike"> <label for="motorbike">Motorbike</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="train"> <label for="train">Train</label>
		</div>
		
		<div id="indoor" class="family-col">
			<input class="category" type="checkbox" id="male" name="gender" value="bottle"> <label for="bottle">Bottle</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="chair"> <label for="chair">Chair</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="diningtable"> <label for="diningtable">Dining table</label><br>
			<input class="category" type="checkbox" id="male" name="gender" value="pottedplant"> <label for="pottedplant">Potted plant</label><br>
			<input class="category" type="checkbox" id="female" name="gender" value="sofa"> <label for="sofa">Sofa</label><br>
			<input class="category" type="checkbox" id="other" name="gender" value="tvmonitor"> <label for="tvmonitor">TV monitor</label>
		</div>

	</div>

	<input id="submit" type="button" value="Submit">

</form>
<script>
function retrieveCategories () {
	var cboxes = document.querySelectorAll('.category');
	var categories = [];
	for (var i = 0; i < cboxes.length; i++) {
		if (cboxes[i].checked) {
			categories.push(cboxes[i].value);
		}
	}
	return categories
} 

var categories = []
submit.onclick = async function (event) {
	categories = retrieveCategories();
    const result = await google.colab.kernel.invokeFunction(
        'update_categories', // The callback name.
        categories, // The arguments.
        {});

}
</script>
</form>
</body>
""")
