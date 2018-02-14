"""Create a Deep Convolutional Generative Adversarial Network."""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf
from preprocess import Dataset
from scipy.io import loadmat


class DCGAN(object):
    """DCGAN object."""

    def __init__(self, dataset, real_size=(32, 32, 3), z_size=100, lr=0.0002,
                 alpha=0.2, beta1=0.5):
        """Initialize network."""
        self.dataset = dataset
        self.z_size = z_size
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1

        self.input_real, self.input_z = self._model_inputs(real_size, z_size)
        self.d_loss, self.g_loss = self._model_loss(self.input_real,
                                                    self.input_z, real_size[2],
                                                    alpha=alpha)
        self.d_opt, self.g_opt = self._optimize(self.d_loss, self.g_loss, lr,
                                                beta1)

    def _model_inputs(self, real_dim, z_dim):
        """Create inputs for generator and discriminator."""
        inputs_real = tf.placeholder(tf.float32, (None, *real_dim),
                                     name="input_real")
        inputs_z = tf.placeholder(tf.float32, (None, z_dim),
                                  name="input_z")

        return inputs_real, inputs_z

    def _cnn_generator(self, z, output_dim, alpha, reuse=False, training=True):
        """Build generator CNN."""
        with tf.variable_scope("generator", reuse=reuse):
            x1 = tf.layers.dense(z, 4 * 4 * 512)
            x1 = tf.reshape(x1, (-1, 4, 4, 512))
            x1 = tf.layers.batch_normalization(x1, training=training)
            x1 = tf.nn.leaky_relu(x1, alpha=alpha)

            x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2,
                                            padding="same")
            x2 = tf.layers.batch_normalization(x2, training=training)
            x2 = tf.nn.leaky_relu(x2, alpha=alpha)

            x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2,
                                            padding="same")
            x3 = tf.layers.batch_normalization(x3, training=training)
            x3 = tf.nn.leaky_relu(x3, alpha=alpha)

            logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2,
                                                padding="same")

            out = tf.tanh(logits)

            return out

    def _cnn_discriminator(self, x, alpha, reuse=False):
        """Build discriminator CNN."""
        with tf.variable_scope("discriminator", reuse=reuse):
            x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding="same")
            x1 = tf.nn.leaky_relu(x1, alpha=alpha)

            x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding="same")
            x2 = tf.layers.batch_normalization(x2, training=True)
            x2 = tf.nn.leaky_relu(x2, alpha=alpha)

            x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding="same")
            x3 = tf.layers.batch_normalization(x3, training=True)
            x3 = tf.nn.leaky_relu(x3, alpha=alpha)

            flat = tf.reshape(x3, (-1, 4 * 4 * 256))
            logits = tf.layers.dense(flat, 1)
            out = tf.sigmoid(logits)

            return out, logits

    def _model_loss(self, input_real, input_z, output_dim, alpha):
        """Get the loss for the discriminator and generator."""
        g_model = self._cnn_generator(input_z, output_dim, alpha=alpha)
        d_mod_real, d_logits_real = self._cnn_discriminator(input_real,
                                                            alpha=alpha)
        d_mod_fake, d_logits_fake = self._cnn_discriminator(g_model,
                                                            reuse=True,
                                                            alpha=alpha)
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                    labels=tf.ones_like(d_mod_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.zeros_like(d_mod_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.ones_like(d_mod_fake)))
        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss

    def _optimize(self, d_loss, g_loss, lr, beta1):
        """Build optimization operations."""
        t_vars = tf.trainable_variables()
        d_vars = [v for v in t_vars if v.name.startswith("discriminator")]
        g_vars = [v for v in t_vars if v.name.startswith("generator")]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_opt = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss,
                                                                     var_list=d_vars)
            g_opt = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss,
                                                                     var_list=g_vars)

        return d_opt, g_opt

    def _pretty_print(self, e, epochs, loss_d, loss_g):
        print("Epoch {}/{}...".format(e + 1, epochs),
              "Discriminator Loss: {:.4f}...".format(loss_d),
              "Generator Loss: {:.4f}".format(loss_g))

    def _serialize_data(self, samples, losses):
        """Save generated images and loss data for visualization."""
        with open('train_samples.pkl', 'wb') as f:
            pkl.dump(samples, f)
        with open('train_loss.pkl', 'wb') as f:
            pkl.dump(losses, f)

    def train(self, epochs=25, batch_size=128, print_every=10,
              show_every=100, figsize=(5, 5)):
        """Train network and save samples."""
        saver = tf.train.Saver()
        sample_z = np.random.uniform(-1, 1, size=(72, self.z_size))

        samples, losses = [], []
        steps = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                for x, y in self.dataset.batches(batch_size):
                    steps += 1

                    batch_z = np.random.uniform(-1, 1, size=(batch_size, self.z_size))

                    _ = sess.run(self.d_opt, feed_dict={self.input_real: x,
                                                        self.input_z: batch_z})
                    _ = sess.run(self.g_opt, feed_dict={self.input_z: batch_z,
                                                        self.input_real: x})

                    if steps % print_every == 0:
                        train_loss_d = self.d_loss.eval({self.input_z: batch_z,
                                                         self.input_real: x})
                        train_loss_g = self.g_loss.eval({self.input_z: batch_z})

                        self._pretty_print(e, epochs, train_loss_d, train_loss_g)
                        losses.append((train_loss_d, train_loss_g))

                    if steps % show_every == 0:
                        gen_samples = sess.run(self._cnn_generator(self.input_z,
                                                                   3, reuse=True,
                                                                   training=False),
                                               feed_dict={self.input_z: sample_z})
                        samples.append(gen_samples)
                        _ = DCGAN.view_samples(-1, samples, 6, 12,
                                               figsize=(figsize))
                        plt.show()

            saver.save(sess, "./checkpoints/generator.ckpt")

        self._serialize_data(samples, losses)

        return losses, samples

    @staticmethod
    def load_pickle_data():
        """Load generated images and loss data from pickle file."""
        with open('train_samples.pkl', 'rb') as f:
            samples = pkl.load(f)
        with open('train_loss.pkl', 'rb') as f:
            losses = pkl.load(f)

        return samples, losses

    @staticmethod
    def view_samples(epoch, samples, nrows=5, ncols=5, figsize=(5, 5)):
        """asdf."""
        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                                 sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch]):
            ax.axis('off')
            img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
            ax.set_adjustable('box-forced')
            ax.imshow(img, aspect='equal')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    @staticmethod
    def visualize_loss(losses):
        """Visualize performance of generator and discriminator."""
        fig, ax = plt.subplots()
        losses = np.array(losses)
        plt.plot(losses.T[0], label="Discriminator")
        plt.plot(losses.T[1], label="Generator")
        plt.title("Training Losses")
        plt.show()


def main():
    """Load data, train network, visualize results."""
    data_dir = 'data/'
    trainset = loadmat(data_dir + 'train_32x32.mat')
    testset = loadmat(data_dir + 'test_32x32.mat')
    dataset = Dataset(trainset, testset)

    tf.reset_default_graph()
    dcgan = DCGAN(dataset)

    losses, samples = dcgan.train()

    # samples, losses = dcgan.load_pickle_data()

    dcgan.view_samples(-1, samples)
    dcgan.visualize_loss(losses)

if __name__ == "__main__":
    main()
