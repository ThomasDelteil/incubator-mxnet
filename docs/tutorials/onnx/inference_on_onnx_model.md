
# Running inference on MXNet from an ONNX model

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

In this tutorial we will:
    
    - learn how to load a pre-trained .onnx model file into MXNet/Gluon
    - learn how to test this model using the sample input/output
    - learn how to test the model on custom images

## Pre-requisite

To run the tutorial you will need to have installed the following python modules:
- [MXNet](http://mxnet.incubator.apache.org/install/index.html)
- [onnx](https://github.com/onnx/onnx)
- [onnx-mxnet](https://github.com/onnx/onnx-mxnet)
- matplotlib
- wget


```python
import numpy as np
import onnx_mxnet
import mxnet as mx
from mxnet import gluon, nd
%matplotlib inline
import matplotlib.pyplot as plt
import tarfile, os
import wget
import json
```

### Downloading supporting files
These are images and a vizualisation script


```python
image_folder = "images"
utils_file = "utils.py" # contain utils function to plot nice visualization
images = ['apron', 'hammerheadshark', 'dog', 'wrench', 'dolphin', 'lotus']
base_url = "https://github.com/ThomasDelteil/web-data/blob/c77c2e93ba142f45682ed63c191d2568b20aff25/mxnet/doc/tutorials/onnx/{}?raw=true"

if not os.path.isdir(image_folder):
    os.makedirs(image_folder)
    for image in images: wget.download(base_url.format("{}/{}.jpg".format(image_folder, image)), image_folder)
if not os.path.isfile(utils_file):
    wget.download(base_url.format(utils_file))                               
```


```python
from utils import *
```

## Downloading a model from the [ONNX model zoo](https://github.com/onnx/models)

We download a pre-trained model, in our case the [vgg16](https://arxiv.org/abs/1409.1556) model, trained on ImageNet. The model comes packaged in a archive `tar.gz` file containing an `model.onnx` model file and some sample input/output data.


```python
base_url = "https://s3.amazonaws.com/download.onnx/models/" 
current_model = "vgg16"
model_folder = "model"
archive = "{}.tar.gz".format(current_model)
archive_file = os.path.join(model_folder, archive)
url = "{}{}".format(base_url, archive)
```

Create the model folder and download the zipped model


```python
os.makedirs(model_folder, exist_ok=True)
if not os.path.isfile(archive_file):  
    wget.download(url, model_folder)
```

Extract the model


```python
if not os.path.isdir(os.path.join(model_folder, current_model)):
    tar = tarfile.open(archive_file, "r:gz")
    tar.extractall(model_folder)
    tar.close()
```

The models have been pre-trained on ImageNet, let's load the label mapping


```python
categories = json.load(open('ImageNet_labels.json', 'r'))
```

## Loading the model into MXNet Gluon


```python
onnx_path = os.path.join(model_folder, current_model, "model.onnx")
```

We get the symbol and parameter objects


```python
sym, params = onnx_mxnet.import_model(onnx_path)
```

We pick a context, CPU or GPU


```python
ctx = mx.cpu()
```

And load them into a MXNet Gluon symbol block


```python
net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('input_0'))
net_params = net.collect_params()
for param in params:
    if param in net_params:
        net_params[param]._load_init(params[param], ctx=ctx)
```


```python
# This hybridize the network so we get performance gains
net.hybridize()
```

## Test using sample inputs and outputs
The model comes with sample input/output we can use to test that the model is correctly loaded


```python
numpy_path = os.path.join(model_folder, current_model, 'test_data_0.npz')
sample = np.load(numpy_path, encoding='bytes')
inputs = sample['inputs']
outputs = sample['outputs']
```


```python
print("Input format: {}".format(inputs[0].shape))
```

    Input format: (1, 3, 224, 224) <!--no-notebook-->


We can visualize the network (requires graphviz installed)


```python
mx.visualization.plot_network(sym, shape={"input_0":inputs[0].shape}, node_attrs={"shape":"oval","fixedsize":"false"})
```




![png](https://github.com/ThomasDelteil/web-data/blob/c77c2e93ba142f45682ed63c191d2568b20aff25/mxnet/doc/tutorials/onnx/network.png?raw=true)<!--no-notebook-->



Helper function to run a batch of data through the net and collate the outputs


```python
def run_batch(net, data):
    results = []
    for batch in data:
        outputs = net(batch)
        for output in outputs.asnumpy(): results.append(output)
    return np.array(results)
```


```python
result = run_batch(net, nd.array(inputs, ctx))
```


```python
print("Loaded model and sample output predict the same class: {}".format(np.argmax(result) == np.argmax(outputs[0])))
```

    Loaded model and sample output predict the same class: True <!--no-notebook-->


Good, now we can run against real data

## Test using real images


```python
TOP_N = 3 # How many top guesses we show
```


```python
# Transform to put the data into a network acceptable format
transform = lambda img: np.expand_dims(np.transpose(img, (2,0,1)),axis=0).astype(np.float32)
```


```python
img0 = plt.imread('images/apron.jpg')
img1 = plt.imread('images/hammerheadshark.jpg')
img2 = plt.imread('images/dog.jpg')
img3 = plt.imread('images/wrench.jpg')
img4 = plt.imread('images/dolphin.jpg')
img5 = plt.imread('images/lotus.jpg')

image_net_images = [img0, img1, img2]
caltech101_images = [img3, img4, img5]
images = image_net_images + caltech101_images
```


```python
batch = nd.array(np.concatenate([transform(img) for img in images], axis=0), ctx=ctx)
result = run_batch(net, [batch])
```


```python
plot_predictions(image_net_images, result[:3], categories, TOP_N)
```


![png](https://github.com/ThomasDelteil/web-data/blob/c77c2e93ba142f45682ed63c191d2568b20aff25/mxnet/doc/tutorials/onnx/imagenet.png?raw=true)<!--no-notebook-->


**Well done!** Looks like it is doing a pretty good job at classifying pictures when the category is in the ImageNet list

Let's now see the results on the other images


```python
plot_predictions(caltech101_images, result[3:7], categories, TOP_N)
```


![png](https://github.com/ThomasDelteil/web-data/blob/c77c2e93ba142f45682ed63c191d2568b20aff25/mxnet/doc/tutorials/onnx/caltech101.png?raw=true)<!--no-notebook-->


**Hmm, not so good...** We see where the network is coming from but effectively `wrench`, `dolphin` and `lotus` categories are not in the ImageNet classes...

Lucky for us, the [Caltech101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) has them, let's see if we can fine-tune our network to classify those

We show that in our next tutorials:
    - [Fine-tuning a ONNX Model using the modern MXNet/Gluon API](addlink)
    - [Fine-tuning a ONNX Model using the old MXNet/Module API](addlink)
    
    <!-- INSERT SOURCE DOWNLOAD BUTTONS -->
