"""Create a GAN that produces MNIST-like images."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


class GanMNIST(object):
    """
    GAN that generates MNIST images.

    Args
        sess (Session): TensorFlow session environment
        mnist (Datasets): MNIST dataset
        input_size (int): input to discriminator, 28x28 MNIST flattened
        z_size (int): latent to generator
        g_hidden (int): generator hidden layer size
        d_hidden (int): discriminator hidden layer size
        alpha (float): leaky ReLU leak factor
        smooth (float): label smoothing factor
        lr (float): learning rate
        epochs (int): # of times data fully processed
        batch_size (int): mini-batch size
    """

    def __init__(self, sess):
        """Initialize GanMNIST object."""
        self.sess = sess
        self.mnist = input_data.read_data_sets("MNIST_data")

        # Hyperparameters
        self.input_size = 784
        self.z_size = 100
        self.g_hidden_size = 128
        self.d_hidden_size = 128
        self.alpha = 0.01
        self.smooth = 0.1
        self.lr = 0.002
        self.epochs = 100
        self.batch_size = 100

        self.build_model()

    def model_inputs(self):
        """Create inputs for generator and discriminator."""
        inputs_real = tf.placeholder(tf.float32, (None, self.input_size),
                                     name="input_real")
        inputs_z = tf.placeholder(tf.float32, (None, self.z_size),
                                  name="input_z")

        return inputs_real, inputs_z

    def generator(self, z, reuse=False):
        """
        Build the generator network.

        Args
            z (Tensor): input tensor for generator
        Returns
            out (Tensor): input tensor for discriminator
        """
        with tf.variable_scope("generator", reuse=reuse):

            h1 = tf.layers.dense(z, self.g_hidden_size, activation=None)
            h1 = tf.nn.leaky_relu(h1, alpha=self.alpha)

            logits = tf.layers.dense(h1, self.input_size, activation=None)
            out = tf.tanh(logits)

            return out

    def discriminator(self, x, reuse=False):
        """
        Build the discriminator network.

        Args
            x (Tensor): input tensor for discriminator
        Returns
            out (Tensor): 1 = real, 0 = fake
            logits (Tensor): log likelihoods for loss function input
        """
        with tf.variable_scope("discriminator", reuse=reuse):

            h1 = tf.layers.dense(x, self.d_hidden_size, activation=None)
            h1 = tf.nn.leaky_relu(h1, alpha=self.alpha)

            logits = tf.layers.dense(h1, 1, activation=None)
            out = tf.sigmoid(logits)

            return out, logits

    def build_model(self):
        """Build network and set Tensors to class attributes."""
        self.input_x, self.input_z = self.model_inputs()
        self.g_model = self.generator(self.input_z)
        self.d_model_real, self.d_logit_real = self.discriminator(self.input_x)
        self.d_model_fake, self.d_logit_fake = self.discriminator(self.g_model,
                                                                  reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_real,
                                                    labels=tf.ones_like(self.d_logit_real) * (1 - self.smooth)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                    labels=tf.zeros_like(self.d_logit_fake)))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
                                                    labels=tf.ones_like(self.d_logit_fake)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars
                       if var.name.startswith('discriminator')]
        self.g_vars = [var for var in t_vars
                       if var.name.startswith('generator')]

        self.saver = tf.train.Saver(var_list=self.g_vars)
        
    def train(self):
        """Train the network and save samples."""
        d_train_op = tf.train.AdamOptimizer().minimize(self.d_loss,
                                                       var_list=self.d_vars)
        g_train_op = tf.train.AdamOptimizer().minimize(self.g_loss,
                                                       var_list=self.g_vars)
        self.sess.run(tf.global_variables_initializer())

        samples = []
        losses = []
        start_time = time.time()

        for e in range(self.epochs):
            for ii in range(self.mnist.train.num_examples // self.batch_size):
                batch = self.mnist.train.next_batch(self.batch_size)

                # Get images, reshape, and rescale for discriminator
                batch_images = batch[0].reshape(self.batch_size, self.input_size)
                batch_images = batch_images * 2 - 1

                # Sample random noise for generator
                batch_z = np.random.uniform(-1, 1, size=(self.batch_size,
                                                         self.z_size))

                # Run optimizers
                _ = self.sess.run(d_train_op, feed_dict={"input_real:0": batch_images,
                                                         "input_z:0": batch_z})
                _ = self.sess.run(g_train_op, feed_dict={"input_z:0": batch_z})

            train_loss_d = self.sess.run(self.d_loss, feed_dict={"input_z:0": batch_z,
                                                                 "input_real:0": batch_images})
            train_loss_g = self.g_loss.eval({"input_z:0": batch_z})

            self._pretty_print(e, self.epochs, train_loss_d, train_loss_g)
            losses.append((train_loss_d, train_loss_g))

            sample_z = np.random.uniform(-1, 1, size=(16, self.z_size))
            gen_samples = self.sess.run(self.generator(self.input_z, reuse=True),
                                        feed_dict={"input_z:0": sample_z})
            samples.append(gen_samples)
            self.saver.save(self.sess, './checkpoints/generator.ckpt')

        end_time = time.time()
        print("Total time elapsed: {:.2f} seconds".format(end_time - start_time))

        self._serialize_data(samples, losses)

    def _pretty_print(self, e, epochs, loss_d, loss_g):
        print("Epoch {}/{}...".format(e + 1, epochs),
              "Discriminator Loss: {:.4f}...".format(loss_d),
              "Generator Loss: {:.4f}".format(loss_g))

    def _serialize_data(self, samples, losses):
        """Save generated images and loss data for visualization."""
        with open('train_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
        with open('train_loss.pkl', 'wb') as f:
            pickle.dump(losses, f)

    @staticmethod
    def load_data():
        """Load generated images and loss data from pickle file."""
        with open('train_samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        with open('train_loss.pkl', 'rb') as f:
            losses = pickle.load(f)

        return samples, losses

    @staticmethod
    def visualize_loss(losses):
        """Visualize performance of generator and discriminator."""
        fig, ax = plt.subplots()
        losses = np.array(losses)
        plt.plot(losses.T[0], label="Discriminator")
        plt.plot(losses.T[1], label="Generator")
        plt.title("Training Losses")
        plt.show()

    @staticmethod
    def view_samples(epoch, samples):
        """Visualize generated MNIST images."""
        fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4,
                                 sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28, 28)), cmap="Greys_r")

        plt.show()


def main():
    """Build and train the network, then output the generated images."""
    tf.reset_default_graph()

    with tf.Session() as sess:
        gan = GanMNIST(sess)
        gan.train()

    samples, losses = GanMNIST.load_data()
    GanMNIST.visualize_loss(losses)
    GanMNIST.view_samples(-1, samples)

if __name__ == "__main__":
    main()
