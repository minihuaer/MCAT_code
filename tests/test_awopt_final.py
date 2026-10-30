import unittest
from dataclasses import replace

import numpy as np

from awopt_final import AWOptConfig, AdaptiveWristOptimizer, constrained_solve, depth_linearization


class PaperAWOptTests(unittest.TestCase):
    def test_gradient_against_finite_differences(self):
        beta = np.array([.7, -1.8])
        g, j, b = depth_linearization(beta, .3, .2, .9)
        for i in range(2):
            delta = np.eye(2)[i] * 1e-6
            gp = depth_linearization(beta + delta, .3, .2, .9)[0]
            gm = depth_linearization(beta - delta, .3, .2, .9)[0]
            self.assertAlmostEqual(j[i], (gp - gm) / 2e-6, places=8)
        self.assertAlmostEqual(j @ beta - b, g - .9)

    def test_matches_explicit_paper_matrix_and_current_output(self):
        opt = AdaptiveWristOptimizer(replace(AWOptConfig(), output_window=1))
        for z in np.linspace(-.7, .7, 20):
            elbow = np.array([0., 0., 10. + z])
            wrist = elbow + [200 * np.sqrt(1-z*z), 0., 200*z/1.3]
            previous, previous_out = opt.beta.copy(), opt.previous_output
            out, info = opt.update([0, 0, 0], elbow, wrist)
            samples = np.asarray(opt.samples)
            F = np.c_[samples[:, 0], samples[:, 0]**2]
            P = np.eye(len(F)) - np.ones((len(F), len(F))) / len(F)
            S, c = F.T @ P @ F, F.T @ P @ samples[:, 1]
            self.assertAlmostEqual(info['min_eigenvalue'], np.linalg.eigvalsh(S)[0])
            if info['updated']:
                _, j, b = depth_linearization(previous, samples[-1, 0], elbow[2]/opt.L, previous_out[2]/opt.L)
                A = S + opt.config.regularization * np.eye(2) + opt.config.depth_regularization * np.outer(j, j)
                rhs = c + opt.config.regularization * previous + opt.config.depth_regularization * b * j
                np.testing.assert_allclose(opt.beta, np.linalg.solve(A, rhs), atol=1e-10)
            self.assertAlmostEqual(out[2], elbow[2] + info['scale']*(wrist[2]-elbow[2]) + opt.L*info['offset'])

    def test_unexcited_windows_hold_parameters(self):
        for levels in ([.2], [-.5, .5]):
            opt = AdaptiveWristOptimizer()
            for i in range(30):
                z = levels[i % len(levels)]
                _, info = opt.update([0, 0, 0], [0, 0, 0], [100, 0, z*100])
                self.assertFalse(info['updated'])
                np.testing.assert_array_equal(opt.beta, [0., -1.])

    def test_recovers_known_affine_distortion(self):
        opt = AdaptiveWristOptimizer(replace(AWOptConfig(), depth_regularization=0, output_window=1))
        errors = []
        for t in range(500):
            z = .75 * np.sin(t / 11)
            truth = np.array([250*np.sqrt(1-z*z), 0, 250*z])
            measured = truth.copy()
            measured[2] = (truth[2] - 30) / 1.4
            out, _ = opt.update([0, 0, 0], [0, 0, 0], measured)
            errors.append(np.linalg.norm(out-truth))
        self.assertLess(max(errors[-100:]), 1e-7)

    def test_motion_and_filter_match_history_formula(self):
        opt = AdaptiveWristOptimizer(replace(AWOptConfig(), estimation_window=2, output_window=7))
        shoulders, elbows, wrists = [], [], []
        for t in range(12):
            shoulders.append(np.array([t*t, t, 0.]))
            elbows.append(np.array([2*t, t*t, 0.]))
            wrists.append(elbows[-1] + [10., 20., 30.])
            out, info = opt.update(shoulders[-1], elbows[-1], wrists[-1])
            m = sum(np.exp(-k*opt.config.shoulder_decay)*np.linalg.norm(shoulders[t-k][:2]-shoulders[t-k-1][:2])
                    + np.exp(-k*opt.config.elbow_decay)*np.linalg.norm(elbows[t-k][:2]-elbows[t-k-1][:2])
                    for k in range(min(5, t)))
            self.assertAlmostEqual(info['motion'], m)
            rho = opt.config.rho_min + (opt.config.rho_max-opt.config.rho_min)*np.tanh(opt.config.motion_sensitivity*m)
            w = np.exp(-rho*np.arange(min(t+1, 7)))
            expected = w @ np.asarray(wrists[::-1][:7]) / w.sum()
            np.testing.assert_allclose(out, expected)

    def test_constraint_is_quadratic_minimum(self):
        A = np.array([[2., .8], [.8, 1.]])
        for rhs, boundary, sign in ((np.array([2., -10.]), -4., 1),
                                    (np.array([2., 10.]), -.25, -1)):
            beta = constrained_solve(A, rhs, AWOptConfig())
            self.assertEqual(beta[1], boundary)
            self.assertAlmostEqual((A @ beta - rhs)[0], 0.)
            self.assertGreaterEqual(sign * (A @ beta - rhs)[1], 0.)

    def test_normalization_invalid_frame_and_reset(self):
        opt = AdaptiveWristOptimizer()
        opt.update([0, 0, 0], [0, 0, 0], [3, 0, 4])
        opt.update([0, 0, 0], [0, 0, 0], [6, 0, 8])
        self.assertEqual(opt.L, 5.)
        with self.assertRaises(ValueError):
            opt.update([0, 0, 0], [0, 0, 0], [np.nan, 0, 0])
        self.assertEqual(len(opt.samples), 2)
        opt.reset()
        self.assertIsNone(opt.L)
        self.assertIsNone(opt.previous_output)
        self.assertEqual(len(opt.motion), 0)


if __name__ == '__main__':
    unittest.main()
