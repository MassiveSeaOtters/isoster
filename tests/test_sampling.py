import numpy as np
import unittest
from isoster.sampling import get_elliptical_coordinates, extract_isophote_data

class TestSampling(unittest.TestCase):
    def test_get_elliptical_coordinates(self):
        # Test circular
        x0, y0 = 50.0, 50.0
        pa, eps = 0.0, 0.0
        
        # Point at (60, 50) should have sma=10, phi=0
        sma, phi = get_elliptical_coordinates(60.0, 50.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, 0.0)
        
        # Point at (50, 60) should have sma=10, phi=pi/2
        sma, phi = get_elliptical_coordinates(50.0, 60.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, np.pi/2)

        # Test elliptical (eps=0.5 -> b=a/2)
        eps = 0.5
        # Point at (50, 55) should have sma=10
        sma, phi = get_elliptical_coordinates(50.0, 55.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, np.pi/2)

    def test_extract_isophote_data(self):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 10.0
        
        x0, y0 = 50.0, 50.0
        sma = 2.0
        eps = 0.0
        pa = 0.0
        
        phi, intens, radii = extract_isophote_data(image, None, x0, y0, sma, eps, pa)
        
        self.assertTrue(len(phi) > 0)
        self.assertTrue(np.allclose(intens, 10.0))
        self.assertTrue(np.all(radii == sma))
        
        # Test out of bounds
        phi, intens, radii = extract_isophote_data(image, None, -10, -10, sma, eps, pa)
        self.assertEqual(len(phi), 0)

if __name__ == '__main__':
    unittest.main()
