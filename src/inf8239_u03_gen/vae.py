"""VAE y utilidades de AE reutilizables (misma lógica que scripts/train_ae_vae.py)."""
from __future__ import annotations


def make_vae_class(tf):
    """Devuelve la clase VAE (Keras) para no importar TensorFlow al cargar el módulo."""

    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
            self.kl_tracker = tf.keras.metrics.Mean(name="kl")

        @property
        def metrics(self):
            return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

        def train_step(self, data):
            images = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as tape:
                mean, log_variance = self.encoder(images, training=True)
                epsilon = tf.random.normal(tf.shape(mean))
                latent = mean + tf.exp(0.5 * log_variance) * epsilon
                reconstruction = self.decoder(latent, training=True)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2))
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss
            gradients = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_tracker.update_state(total_loss)
            self.reconstruction_tracker.update_state(reconstruction_loss)
            self.kl_tracker.update_state(kl_loss)
            return {metric.name: metric.result() for metric in self.metrics}

    return VAE


def split_autoencoder(tf, autoencoder):
    """Separa un AE de build_autoencoder en (encoder, decoder) para poder muestrear.

    El AE base es Flatten -> Dense(latente) -> Dense(784) -> Reshape. Se reutilizan
    las MISMAS capas (con sus pesos entrenados), no copias.
    """
    latent_layer = autoencoder.get_layer("latent")
    encoder = tf.keras.Model(autoencoder.input, latent_layer.output, name="ae_encoder")
    dense_layers = [layer for layer in autoencoder.layers if isinstance(layer, tf.keras.layers.Dense)]
    reshape_layers = [layer for layer in autoencoder.layers if isinstance(layer, tf.keras.layers.Reshape)]
    latent_dim = latent_layer.units
    latent_inputs = tf.keras.Input((latent_dim,))
    outputs = reshape_layers[-1](dense_layers[-1](latent_inputs))
    decoder = tf.keras.Model(latent_inputs, outputs, name="ae_decoder")
    return encoder, decoder
