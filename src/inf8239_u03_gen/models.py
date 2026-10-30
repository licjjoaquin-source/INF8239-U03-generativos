from __future__ import annotations


def build_autoencoder(tf, latent_dim: int = 16):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(values)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(latent)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    return tf.keras.Model(inputs, outputs, name="autoencoder")


def build_encoder_decoder(tf, latent_dim: int = 2):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    values = tf.keras.layers.Dense(128, activation="relu")(values)
    mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(values)
    log_variance = tf.keras.layers.Dense(latent_dim, name="z_log_var")(values)
    encoder = tf.keras.Model(inputs, [mean, log_variance], name="encoder")
    latent_inputs = tf.keras.Input((latent_dim,))
    values = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(values)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
    return encoder, decoder
