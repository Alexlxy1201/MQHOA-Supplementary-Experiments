import unittest
import numpy as np
from workflow import Operators,configuration,specification,execute,check_numeric

class OpticalTests(unittest.TestCase):
    def test_zero_distance_and_plane_wave(self):
        cfg=configuration()
        cfg['distance_step_m']=0.0
        e=Operators(32,cfg)
        aperture=e.run(dict(op='aperture'),[])
        spectrum=e.run(dict(op='spectrum'),[aperture])
        actual=e.run(dict(op='propagate',plane=0),[spectrum])
        np.testing.assert_allclose(actual,aperture,rtol=1e-12,atol=1e-12)
        # A constant field on the whole padded domain has only DC Fourier power.
        cfg['distance_step_m']=0.003
        full=np.ones((64,64),dtype=np.complex128)
        actual=e.run(dict(op='propagate',plane=0),[np.fft.fft2(full)])
        expected=np.full((32,32),np.exp(1j*2*np.pi/cfg['wavelength_m']*0.003))
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)

    def test_graph_and_metrics(self):
        cfg=configuration()
        case=dict(id='test',n=32,planes=3)
        tasks=specification(case)
        self.assertEqual(len(tasks),12)
        e,outputs=execute(case,cfg)
        for t in tasks:
            self.assertTrue(all(i<t['id'] for i in t['inputs']))
            self.assertTrue(np.all(np.isfinite(outputs[t['id']])))
        final=outputs[len(tasks)-1]
        self.assertEqual(final.shape,(13,))
        for p in range(3):
            np.testing.assert_array_equal(final[4*p:4*p+4],outputs[4+3*p])
            self.assertGreater(final[4*p],0)
            self.assertTrue(0<=final[4*p+2]<=1)
        self.assertEqual(final[-1],np.argmin(final[:-1].reshape(-1,4)[:,3]))

    def test_numeric_gate_rejects_corruption(self):
        cfg=configuration()
        with self.assertRaises(RuntimeError):
            check_numeric(np.ones(4),np.zeros(4),cfg)

if __name__=='__main__':
    unittest.main(verbosity=2)
