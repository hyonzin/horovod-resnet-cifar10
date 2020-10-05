# ResNet: Classifying CIFAR-10 images *with Horovod*

**Note**: this project is extension of tensorflow project [github.com/chao-ji/tf-resnet-cifar10](https://www.github.com/chao-ji/tf-resnet-cifar10)

* A lightweight TensorFlow implementation of ResNet model for classifying CIFAR-10 images. 

* Reproduces the results presented in the paper.

* Shows the full schematic diagram of a 20-layer ResNet annotated with feature map sizes.

### Usage
##### Clone the Repo
```
git clone git@github.com:hyonzin/horovod-resnet-cifar10.git
```
##### Download and untar CIFAR-10 dataset
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xvzf cifar-10-binary.tar.gz
```
##### Train the ResNet Classifier
To train the ResNet model using default settings, simply run
```
python run_trainer.py \
  --data_path=cifar-10-batches-bin \
  --num_layers=110
```
To change the number of layers in the ResNet (for example, to 110), specify `--num_layers=110`. To degenerate the ResNet model to a *Plain network*, specify `--shortcut_connections=False`. To see a full list of arguments, run
```
python run_trainer.py --help
```
##### Evaluate a Trained ResNet Classifier
To evaluate the trained model on the test set (10,000 images), run
```
  python run_evaluator.py \
    --path=cifar-10-batches-bin \
    --ckpt_path=/PATH/TO/CKPT \
    --num_layers=110
```
Note that you need to specify the path to the checkpoint file containing trained weights via `--ckpt_path`.

### References:
  1. <a name="myfootnote1">ResNet V1</a>, Deep Residual Learning for Image Recognition, He *et al.*
  2. <a name="myfootnote2">ResNet V2</a>, Identity Mappings in Deep Residual Networks, He *et al.*
