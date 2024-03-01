import tensorflow as tf

class Op(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[6], dtype=tf.float32), tf.TensorSpec(shape=[6], dtype=tf.float32)])
    def __call__(self, x, y):
        return tf.add(x, y)

if __name__ == "__main__":

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(8, 8, 1)),
        # tf.keras.layers.Dense(units=32, use_bias=True, bias_initializer="ones"),
        # tf.keras.layers.Conv2D(2, 3,)# activation="relu"),
        tf.keras.layers.Softmax(),
    ])
    out = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    # op 
    # model = Op()
    # out = tf.lite.TFLiteConverter.from_concrete_functions([model.__call__.get_concrete_function()]).convert()

    with open("./op.tflite", "wb") as f:
        print("CREATING FILE")
        f.write(out)