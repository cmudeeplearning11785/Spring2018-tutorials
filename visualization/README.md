# Visualization Tutorials

Tutorials on Deep Learning

## Inferno and Tensorboard

Inferno provides a convenient wrapper for training, logging, and other boilerplate. It supports logging
to TensorBoard out-of-the-box.

TensorBoard serves log files that are stored on disk. It is not necessary to have TensorBoard running as you train
(start it before, after, whatever).
You can also copy the log files to a different machine and run TensorBoard from the other machine. If you have
multiple log files in different subdirectories, run TensorBoard in the parent directory and it will let you
compare graphs between different log files.

- Start tensorboard server: `tensorboard --logdir .`
- Run the script: `python pytorch_mnist_inferno_tensorboard_example.py`
- Browse to TensorBoard to view live results: `https://localhost:6006`

## TNT and Visdom

TNT provides some wrappers for training and logging (but not a ton). It also supports logging to Visdom but
it requires a decent amount of coding.

Visdom is a server that receives data from your script. It is necessary to have a running Visdom server
before you start your script. It is also necessary to save your Visdom workspace if you want to look at it later.

- Start visdom server: `python -m visdom.server`
- Run the script: `python pytorch_mnist_tnt_visdom_example.py`
- Browse to visdom (it will be a blank until results start being generated): `https://localhost:8097`
