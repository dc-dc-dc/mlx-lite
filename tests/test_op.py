from op_test import compare_tf, generate_op_file
import os
import unittest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class TestOps(unittest.TestCase):
    def test_add(self):
        shapes = [(1, 100, 100), (1, 100, 100)]
        path = generate_op_file("Add", shapes)
        compare_tf(path, shapes)
        os.remove(path)

    def test_sub(self):
        shapes = [(1, 100, 100), (1, 100, 100)]
        path = generate_op_file("Subtract", shapes)
        compare_tf(path, shapes)
        os.remove(path)

if __name__ == "__main__":
    unittest.main()