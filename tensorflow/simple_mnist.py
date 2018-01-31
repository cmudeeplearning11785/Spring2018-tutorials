import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.learn.python.learn.learn_runner import run
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

"""
This simple program provides command-line parsing, resuming training, graphing validation loss on TensorBoard,
and lots of other freebies.

**Overview**
- `model_fn`: is the code function that defines the model, losses, and metrics
- `make_input_fns`: defines the training and validation input
- `experiment_fn`: links the model and the input
- `main`: starts the experiment
"""


# This is the most important section
# Define your model, losses and metrics
def model_fn(features, labels, mode, params):
    # Simple MLP Model
    # Hyperparameters are stored in `params`
    h = features['x']
    h = slim.flatten(h)
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(params.l2),
                        num_outputs=params.hidden_units,
                        activation_fn=tf.nn.leaky_relu):
        for i in range(params.hidden_layers):
            h = slim.fully_connected(h, scope='my_hidden_layer_{}'.format(i))
        logits = slim.fully_connected(h, num_outputs=10, activation_fn=None, scope='my_output_layer')

    # Predictions
    classes = tf.argmax(logits, axis=1)
    predictions = {
        'classes': classes
    }
    # Any `tf.summary` you create will automatically show up on TensorBoard
    accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, labels), tf.float32))
    tf.summary.scalar('accuracy', accuracy)  # add a scalar graph
    tf.summary.histogram('logits', logits)  # add a histogram
    # Loss
    softmax_loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=tf.one_hot(labels, 10, axis=1)
    )
    tf.summary.scalar('softmax_loss', softmax_loss)  # add another graph
    regularization_loss = tf.losses.get_regularization_loss()
    tf.summary.scalar('regularization_loss', regularization_loss)  # add another graph
    loss = softmax_loss + regularization_loss
    # Training
    optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    # Evaluation
    eval_metric_ops = {'eval_accuracy': tf.metrics.accuracy(labels=labels, predictions=classes)}
    # Return everything wrapped in an EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops=eval_metric_ops, predictions=predictions)


# Data loading and massaging (int to float and rescale)
def make_input_fns():
    train, eval = mnist.load_data()
    fns = []
    for data in (train, eval):
        x, y = data
        x = (x.astype(np.float32) * 2. / 255.) - 1.
        y = y.astype(np.int64)
        fns.append(tf.estimator.inputs.numpy_input_fn(
            {'x': x}, y,
            batch_size=tf.flags.FLAGS.batch_size,
            num_epochs=None,
            shuffle=True
        ))
    return fns


# This function links your input function and your model
def experiment_fn(run_config, hparams):
    train_input_fn, eval_input_fn = make_input_fns()
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )
    return experiment


# Ordinary main function
def main(_argv):
    # Pass command-line arguments to RunConfig
    run_config = RunConfig(
        model_dir=tf.flags.FLAGS.model_dir,
        save_checkpoints_steps=tf.flags.FLAGS.save_checkpoints_steps)
    # Default hyperparameters
    hparams = HParams(l2=1e-3, lr=1e-3, hidden_layers=3, hidden_units=200) \
        # Parse the hparams command-line argument
    hparams.parse(tf.flags.FLAGS.hparams)
    # Run the experiment
    run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # Define command line arguments
    tf.flags.DEFINE_string('model_dir', 'demo/simple_mnist', 'Output directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'How often to save and validate')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    # This line will call main()
    tf.app.run()
