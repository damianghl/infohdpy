import unittest
import numpy as np
from info_hdp import InfoHDP
from scipy import stats, special, integrate, optimize
from typing import List, Tuple, Union

class TestInfoHDP(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.p = np.array([0.1, 0.9])
        self.alpha = 0.5
        self.beta = 0.5
        self.Ns = 10
        self.ndist = 1
        self.M = 100

    def test_strue(self):
        result = InfoHDP.strue(self.p)
        expected = -np.sum(np.where(self.p > 0, self.p * np.log(self.p), 0))
        self.assertAlmostEqual(result, expected, places=5)

    def test_sxtrue(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        result = InfoHDP.sxtrue(p)
        p_reshaped = p.reshape(-1, 2)
        expected = InfoHDP.strue(np.sum(p_reshaped, axis=1))
        self.assertAlmostEqual(result, expected, places=5)

    def test_sytrue(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        result = InfoHDP.sytrue(p)
        p_reshaped = p.reshape(-1, 2)
        expected = InfoHDP.strue(np.sum(p_reshaped, axis=0))
        self.assertAlmostEqual(result, expected, places=5)

    def test_itrue(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        result = InfoHDP.itrue(p)
        expected = -InfoHDP.strue(p) + InfoHDP.sytrue(p) + InfoHDP.sxtrue(p)
        self.assertAlmostEqual(result, expected, places=5)

    def test_gen_prior_pij(self):
        result = InfoHDP.gen_prior_pij(self.alpha, self.beta, self.ndist, self.Ns)
        self.assertEqual(result.shape, (self.ndist, 2 * self.Ns))

    def test_gen_samples_prior(self):
        qij = InfoHDP.gen_prior_pij(self.alpha, self.beta, self.ndist, self.Ns)
        result = InfoHDP.gen_samples_prior(qij, self.M, self.Ns)
        self.assertEqual(result.shape, (self.M,))

    def test_dkm2(self):
        sam = np.array([1, 1, 2, 2, 2, 3])
        result = InfoHDP.dkm2(sam)
        expected = [(1, 1), (2, 1), (3, 1)]
        self.assertEqual(result, expected)

    def test_snaive(self):
        sam = np.array([1, 1, 2, 2, 2, 3])
        nn = len(sam)
        dkm2 = InfoHDP.dkm2(sam)
        result = InfoHDP.snaive(nn, dkm2)
        expected = -sum(count * (freq / nn) * np.log(freq / nn) for freq, count in dkm2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_smaxlik(self):
        sam = np.array([1, 1, 2, 2, 2, 3])
        result = InfoHDP.smaxlik(sam)
        nn = len(sam)
        dkm2 = InfoHDP.dkm2(sam)
        expected = InfoHDP.snaive(nn, dkm2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_inaive(self):
        sam = np.array([1, -1, 2, -2, 2, -3])
        result = InfoHDP.inaive(sam)
        nn = len(sam)
        samxz = np.abs(sam)
        samyz = np.sign(sam)
        dkmz = InfoHDP.dkm2(sam)
        dkmzX = InfoHDP.dkm2(samxz)
        dkmzY = InfoHDP.dkm2(samyz)
        expected = (InfoHDP.snaive(nn, dkmzX) + 
                    InfoHDP.snaive(nn, dkmzY) - 
                    InfoHDP.snaive(nn, dkmz))
        self.assertAlmostEqual(result, expected, places=5)

    def test_Smap(self):
        sam = np.array([1, -1, 2, -2, 2, -3])
        result = InfoHDP.Smap(sam)
        nn = len(sam)
        dkmz = InfoHDP.dkm2(sam)
        kz = len(np.unique(sam))
        az = InfoHDP.asol(nn, kz)
        expected = InfoHDP.Spost(az, nn, dkmz)
        self.assertAlmostEqual(result, expected, places=5)

    def test_Sint(self):
        sam = np.array([1, -1, 2, -2, 2, -3])
        result, _ = InfoHDP.Sint(sam)
        nn = len(sam)
        dkmz = InfoHDP.dkm2(sam)
        kz = len(np.unique(sam))
        az = InfoHDP.asol(nn, kz)
        log_az = np.log(az)
        lower_bound = log_az - 3
        upper_bound = log_az + 3

        def integrand_normalization(log_x):
            return np.exp(InfoHDP.logLa(np.exp(log_x), nn, kz))

        def integrand_weighted_spost(log_x):
            x = np.exp(log_x)
            return InfoHDP.Spost(x, nn, dkmz) * np.exp(InfoHDP.logLa(x, nn, kz))

        norm_const, norm_error = integrate.quad(integrand_normalization, lower_bound, upper_bound)
        weighted_integral, weighted_error = integrate.quad(integrand_weighted_spost, lower_bound, upper_bound)
        expected = weighted_integral / norm_const

        self.assertAlmostEqual(result, expected, places=5)

    def test_Insb(self):
        sam = np.array([1, -1, 2, -2, 2, -3])
        result, sx, sy, sxy = InfoHDP.Insb(sam)
        samxz = np.abs(sam)
        samyz = np.sign(sam)
        
        sx_expected, _ = InfoHDP.Sint(samxz)
        sy_expected, _ = InfoHDP.Sint(samyz)
        sxy_expected, _ = InfoHDP.Sint(sam)
        
        insb_expected = sx_expected + sy_expected - sxy_expected
        self.assertAlmostEqual(result, insb_expected, places=5)
        self.assertAlmostEqual(sx, sx_expected, places=5)
        self.assertAlmostEqual(sy, sy_expected, places=5)
        self.assertAlmostEqual(sxy, sxy_expected, places=5)
        
    def test_Spost(self):
        x, nn = 2.0, 100
        dkm = [(1, 3), (2, 1), (3, 1)]
        result = InfoHDP.Spost(x, nn, dkm)
        expected = (special.polygamma(0, nn + x + 1) - 
                    (1 / (x + nn)) * sum(count * freq * special.polygamma(0, freq + 1) for freq, count in dkm))
        self.assertAlmostEqual(result, expected, places=5)

    def test_SYconX(self):
        x, bb, nn = 2.0, 0.5, 100
        n10 = [[5, 10], [15, 20], [25, 30]]
        result = InfoHDP.SYconX(x, bb, nn, n10)
        expected = ((x / (x + nn)) * (special.polygamma(0, 2*bb+1) - special.polygamma(0, bb+1)) + 
                    (1 / (x + nn)) * sum((n1 + n0) * (special.polygamma(0, n1 + n0 + 2*bb + 1) - 
                                                      (n1 + bb) / (n1 + n0 + 2*bb) * special.polygamma(0, n1 + bb + 1) -
                                                      (n0 + bb) / (n1 + n0 + 2*bb) * special.polygamma(0, n0 + bb + 1))
                                         for n1, n0 in n10))
        self.assertAlmostEqual(result, expected, places=5)

    def test_bsol(self):
        kx = 10
        n10 = [[5, 10], [15, 20], [25, 30]]
        result = InfoHDP.bsol(kx, n10)
        
        def objective(log_b):
            return -InfoHDP.logLb(np.exp(log_b), kx, n10)
        
        sol = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        expected = np.exp(sol.x)
        self.assertAlmostEqual(result, expected, places=5)

    def test_logLb(self):
        b, kx = 0.5, 10
        n10 = [[5, 10], [15, 20], [25, 30]]
        result = InfoHDP.logLb(b, kx, n10)
        expected = (kx * (special.gammaln(2*b) - 2*special.gammaln(b)) + 
                    sum(special.gammaln(b + n1) + special.gammaln(b + n0) - special.gammaln(2*b + n1 + n0) 
                        for n1, n0 in n10) +
                    np.log(b) + np.log(2*special.polygamma(1, 2*b+1) - special.polygamma(1, b+1)))
        self.assertAlmostEqual(result, expected, places=5)
        
    def test_IhdpMAP(self):
        sam = np.array([1, -1, 2, -2, 2, -3])
        result = InfoHDP.IhdpMAP(sam)
        nn = len(sam)
        kk = len(np.unique(sam))
        a1 = InfoHDP.asol(nn, kk)
        samx = np.abs(sam)
        kx = len(np.unique(samx))
        n10 = InfoHDP.n10sam(sam)
        b1 = InfoHDP.bsol(kx, n10)
        sy = InfoHDP.smaxlik(np.sign(sam))
        sycx = InfoHDP.SYconX(a1, b1, nn, n10)
        expected = sy - sycx
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == '__main__':
    unittest.main()
