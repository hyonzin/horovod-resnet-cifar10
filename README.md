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

##### Training
To **train** a model, run
```
python run_trainer.py \
  --data_path=/path/to/cifar10/binary/files \
```
Note that you can terminate training prematurely, and pick up where you left off by setting `--ckpt_path=` to the path to the directory containing all checkpoint files generated so far. The parameters will be restored from the most recent checkpoint. Also, the training metrics (loss and accuracy) will be written to `./log`. Run `tensorboard --logdir=log` to view tensorboard.

##### Evaluation
To **evaluate** a model, run
```
python run_evaluator.py \
  --data_path=/path/to/cifar10/binary/files \
  --ckpt_path=/path/to/directory/ckpt/files/will/be/loaded/from
```

To see full list of arguments, run

```
python run_trainer.py --help
python run_evaluator.py --help
```

### References:
  1. <a name="myfootnote1">ResNet V1</a>, Deep Residual Learning for Image Recognition, He *et al.*
  2. <a name="myfootnote2">ResNet V2</a>, Identity Mappings in Deep Residual Networks, He *et al.*
