from op_test import compare_tf, generate_op_file
import os
import unittest
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class TestOps(unittest.TestCase):
    def help_test(self, op_name: str, shapes: List[int], **options):
        path = generate_op_file(op_name, shapes, **options)
        compare_tf(path, shapes)
        os.remove(path)

    def test_add(self):
        self.help_test("Add", [(1, 100, 100), (1, 100, 100)])

    def test_sub(self):
        self.help_test("Subtract", [(1, 100, 100), (1, 100, 100)])

    def test_mul(self):
        self.help_test("Multiply", [(1, 100, 100), (1, 100, 100)])

    def test_average(self):
        self.help_test("Average", [(1, 100, 100), (1, 100, 100)])

    def test_conv2d(self):
        self.help_test("Conv2D", [(1, 100, 100, 3)], filters=3, kernel_size=3)
        self.help_test("Conv2D", [(1, 100, 100, 3)], filters=3, kernel_size=3, strides=2)
    
    def test_maxpooling2d(self):
        self.help_test("MaxPooling2D", [(1, 100, 100, 3)], pool_size=2)
        self.help_test("MaxPooling2D", [(1, 100, 100, 3)], pool_size=2, strides=2)

if __name__ == "__main__":
    unittest.main()