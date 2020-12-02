import tensorflow as tf
import horovod.tensorflow as hvd

from dataset import Cifar10DatasetBuilder 
from dataset import read_data
from model import ResNetCifar10 
from model_runners import ResNetCifar10Trainer
from model_runners import build_optimizer
from absl import flags
from absl import app 


flags.DEFINE_string('data_path', None, 'The path to the directory containing '
                    'Cifar10  binary files.')
flags.DEFINE_string('ckpt_path', '/tmp/resnet/', 'The path to the directory that'
                    ' checkpoints will be written to or loaded from.')
flags.DEFINE_string('log_path', '.', 'The path to the directory to which'
                    'tensorboard log files will be written.')

flags.DEFINE_integer('num_layers', 20, 'Number of weighted layers. Valid '
                     'values: 20, 32, 44, 56, 110')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('num_iterations', 64000, 'The num of training iterations.')
flags.DEFINE_integer('log_per_iterations', 200, 'Every N iterations to save '
                     ' checkpoint file and print training metrics.')
flags.DEFINE_integer('shuffle_buffer_size', 50000, 'The buffer size for '
                     'shuffling the training set.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for the momentum optimizer.')
flags.DEFINE_float('init_lr', 0.1, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, 'Weight decay for l2 regularization.')
flags.DEFINE_float('batch_norm_momentum', 0.99, 'Moving average decay.')
flags.DEFINE_boolean('shortcut_connection', True, 'Whether to add shortcut '
                     'connection. Defaults to True. False for Plain network.')

flags.DEFINE_string('compression', 'NONE', 'Gradient compression method')

flags.DEFINE_float('k', 1.0, 'parameter for Allreduce-Adacomp')

FLAGS = flags.FLAGS


def main(_):
  # Initialize Horovod
  hvd.init()

  # Pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  # Scale the learning rate by the number of workers
  #learning_rate = FLAGS.init_lr * hvd.size()
  learning_rate = FLAGS.init_lr

  compression = hvd.Compression.none
  if FLAGS.compression == "adacompGPU":
      compression = hvd.Compression.adacompGPU
  elif FLAGS.compression == "adacompGPU":
      compression = hvd.Compression.adacompCPU
  elif FLAGS.compression == "fp16":
      compression = hvd.Compression.fp16
  elif FLAGS.compression == "allreduceAdacomp":
      hvd.Compression.allreduceAdacomp.K = FLAGS.k
      compression = hvd.Compression.allreduceAdacomp

  print("compression:", compression)
  #hvd.Compression.allreduceAdacomp.R=20
  #hvd.Compression.allreduceAdacomp.K=100
  
  builder = Cifar10DatasetBuilder(buffer_size=FLAGS.shuffle_buffer_size)
  labels, images = read_data(FLAGS.data_path, training=True)
  dataset = builder.build_dataset(
      labels, images, FLAGS.batch_size, training=True)

  model = ResNetCifar10(FLAGS.num_layers, 
                        shortcut_connection=FLAGS.shortcut_connection, 
                        weight_decay=FLAGS.weight_decay, 
                        batch_norm_momentum=FLAGS.batch_norm_momentum)
  optimizer = build_optimizer(init_lr=learning_rate, momentum=FLAGS.momentum)

  # Wrap the optimizer in hvd.DistributedOptimizer
  optimizer = hvd.DistributedOptimizer(optimizer)

  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

  trainer = ResNetCifar10Trainer(model)
  trainer.train(dataset, 
                optimizer, 
                ckpt, 
                FLAGS.batch_size, 
                FLAGS.num_iterations, 
                FLAGS.log_per_iterations, 
                FLAGS.ckpt_path,
                compression)

if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  app.run(main)
