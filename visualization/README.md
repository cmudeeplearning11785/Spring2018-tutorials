# Visualization Tutorials

Here are two examples of how to visualize pytorch training metrics. These are not the only ways
to visualize your data and not the only frameworks but should give you some good ideas to start with.

- TensorBoard
- Visdom

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

![TensorBoard CMD](https://github.com/cmudeeplearning11785/deep-learning-tutorials/raw/master/visualization/tensorboard-cmd.png)

![TensorBoard CMD](https://github.com/cmudeeplearning11785/deep-learning-tutorials/raw/master/visualization/tensorboard.png)


```python
trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
    .validate_every((2, 'epochs')) \
    .save_every((5, 'epochs')) \
    .save_to_directory(args.save_directory) \
    .set_max_num_epochs(10) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every='never'),
                  log_directory=args.save_directory)
```

## TNT and Visdom

TNT provides some wrappers for training and logging (but not a ton). It also supports logging to Visdom but
it requires a decent amount of coding.

Visdom is a server that receives data from your script. It is necessary to have a running Visdom server
before you start your script. It is also necessary to save your Visdom workspace if you want to look at it later.

- Start visdom server: `python -m visdom.server`
- Run the script: `python pytorch_mnist_tnt_visdom_example.py`
- Browse to visdom (it will be a blank until results start being generated): `https://localhost:8097`

![TensorBoard CMD](https://github.com/cmudeeplearning11785/deep-learning-tutorials/raw/master/visualization/visdom-cmd.png)

![TensorBoard CMD](https://github.com/cmudeeplearning11785/deep-learning-tutorials/raw/master/visualization/visdom.png)

```python
train_loss_logger = VisdomPlotLogger(
    'line', port=port, opts={'title': 'Train Loss'})
train_err_logger = VisdomPlotLogger(
    'line', port=port, opts={'title': 'Train Class Error'})
test_loss_logger = VisdomPlotLogger(
    'line', port=port, opts={'title': 'Test Loss'})
test_err_logger = VisdomPlotLogger(
    'line', port=port, opts={'title': 'Test Class Error'})
...
def on_end_epoch(state):
    print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_err_logger.log(state['epoch'], classerr.value()[0])

    # do validation at the end of each epoch
    reset_meters()
    engine.test(train_fn, validate_loader)
    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_err_logger.log(state['epoch'], classerr.value()[0])
    print('Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
...
engine.hooks['on_end_epoch'] = on_end_epoch
engine.train(train_fn, train_loader, maxepoch=args.epochs, optimizer=optimizer)
```
