# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import operator
import argparse
import numpy as np
import json
import operator
from process_image import process_image
import warnings
warnings.filterwarnings('ignore')
import pprint as pp

#----------------------------------------------------------------------
# Input arguement handling
parser = argparse.ArgumentParser(description='prediction.py: Finds flower name to image')

parser.add_argument('image', 
                    action="store", 
                    help='Image name with path',
                    default='./')

parser.add_argument('model', 
                    action="store", 
                    help='Model name - present at current folder',
                    default='model.h5')

parser.add_argument('--top_k', 
                    action="store", 
                    dest='numlabel',
                    help='K: the K most prob labels are displayed', 
                    required=False,
                    default=5,
                    type= int)

parser.add_argument('--category_names', 
                    action="store", 
                    dest='mapjson',
                    required=False,
                    help='map.json: matching flower name table',
                    default='')

argvals = parser.parse_args()
#print('argvals.ModelName: ', argvals.model)

label_avl = len(argvals.mapjson)>0

'''
print('argvals.image: ', argvals.image)
print('argvals.--top_k: ', argvals.numlabel)
print('argvals.--categorynames: ', argvals.mapjson)
print('type(argvals.image)',type(argvals.image))
print('type(argvals.mapjson)',type(argvals.mapjson))
'''

#--------------------------------------------------------------------
# Loading data

# load image
imagejpg = Image.open(argvals.image)

# load model
model=tf.keras.models.load_model(argvals.model, custom_objects={'KerasLayer':hub.KerasLayer})
#print(model.summary())

# load list
if label_avl:
    with open(argvals.mapjson, 'r') as f:
        class_names = json.load(f)

#--------------------------------------------------------------------
# Preprocess image
npimage = np.asarray(imagejpg)
npimage_rez = process_image(npimage)
npimage_rez_formodel = np.expand_dims(npimage_rez,0)
tfimage_rez_formodel=tf.convert_to_tensor(npimage_rez_formodel)

#--------------------------------------------------------------------
# Inference of image
praed = model.predict(tfimage_rez_formodel)
pylist = praed[0].tolist()

#--------------------------------------------------------------------
# Sort the result
pdict = dict(zip(list(range(102)),pylist))
sortedDict = sorted(pdict.items(), key=operator.itemgetter(1), reverse=True)
probs=[]
classes=[]
for i in range(argvals.numlabel):
    probs.append(sortedDict[i][1])
    classes.append(sortedDict[i][0])

if label_avl:
    strlistlabels=[class_names[str(i+1)] for i in classes]
else:
    strlistlabels=classes
    
#--------------------------------------------------------------------
# Plotting the result
print('-------------------------------------')
print('Image file name is: {}'.format(argvals.image))

# label names avl
if label_avl:
    print('Most probable flower name: {}'.format(strlistlabels[0]))
    print('Most {} probable classes and probabilities'.format(argvals.numlabel))
else:
    print('Most probable flower class number: {}'.format(strlistlabels[0]))
    print('Most {} probable class numbers and probabilities'.format(argvals.numlabel))

#generate dictionry
dict_labels = dict(zip(strlistlabels,probs))
pp.pprint(dict_labels)
