import numpy as np
import unittest
from isoster.model import build_ellipse_model

class TestModel(unittest.TestCase):
    def test_build_ellipse_model_basic(self):
        shape = (100, 100)
        # Create a single isophote
        iso = {
            'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0,
            'intens': 100.0
        }
        
        model = build_ellipse_model(shape, [iso])
        
        # Check center
        self.assertEqual(model[50, 50], 100.0)
        # Check inside radius
        self.assertEqual(model[50, 55], 100.0)
        # Check outside radius
        self.assertEqual(model[50, 70], 0.0)
        
    def test_build_ellipse_model_layered(self):
        shape = (100, 100)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0, 'intens': 100.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 20.0, 'eps': 0.0, 'pa': 0.0, 'intens': 50.0}
        ]
        
        model = build_ellipse_model(shape, isos)
        
        # Inner part should be 100 (overwrite 50)
        self.assertEqual(model[50, 50], 100.0)
        # Middle part should be 50
        self.assertEqual(model[50, 65], 50.0)
        # Outer part should be 0
        self.assertEqual(model[50, 80], 0.0)

if __name__ == '__main__':
    unittest.main()
