import numpy as np
import unittest
from isoster.fitting import fit_first_and_second_harmonics, sigma_clip, compute_aperture_photometry

class TestFitting(unittest.TestCase):
    def test_fit_harmonics(self):
        phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
        y0, A1, B1, A2, B2 = 100.0, 10.0, 5.0, 2.0, 1.0
        intens = y0 + A1*np.sin(phi) + B1*np.cos(phi) + A2*np.sin(2*phi) + B2*np.cos(2*phi)
        
        coeffs, cov = fit_first_and_second_harmonics(phi, intens)
        self.assertTrue(np.allclose(coeffs, [y0, A1, B1, A2, B2], atol=1e-5))

    def test_sigma_clip(self):
        # We need at least 11 points for a single outlier to be clipped at 3rd sigma
        # due to the outlier's own contribution to the standard deviation.
        phi = np.arange(20)
        outlier_val = 1000.0
        intens = np.array([10.0] * 19 + [outlier_val])
        
        # Test no clipping
        p, i, n = sigma_clip(phi, intens, nclip=0)
        self.assertEqual(n, 0)
        self.assertEqual(len(i), 20)
        
        # Test 1 iteration of clipping
        p, i, n = sigma_clip(phi, intens, sclip=3.0, nclip=1)
        self.assertEqual(n, 1)
        self.assertEqual(len(i), 19)
        self.assertNotIn(outlier_val, i)

    def test_aperture_photometry(self):
        image = np.ones((100, 100))
        x0, y0 = 50.0, 50.0
        sma = 10.0
        eps = 0.0
        pa = 0.0
        
        tflux_e, tflux_c, npix_e, npix_c = compute_aperture_photometry(image, None, x0, y0, sma, eps, pa)
        
        # Area of r=10 circle is approx 314
        expected_area = np.pi * sma**2
        self.assertLess(abs(npix_c - expected_area) / expected_area, 0.05)
        self.assertAlmostEqual(tflux_c, npix_c)
        self.assertAlmostEqual(tflux_e, tflux_c)
        self.assertEqual(npix_e, npix_c)

if __name__ == '__main__':
    unittest.main()
