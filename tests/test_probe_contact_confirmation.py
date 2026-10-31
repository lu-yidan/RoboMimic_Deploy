"""Exercise contact dropout, wrong-rod rejection and confirmation timeout.

These are state-machine tests with synthetic sensor readings. Real dynamics
and bounded QP feasibility are checked by the separate simulation and QP tests.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from training.probe_hook_loading import install_hook_admission


class ProbeFixture:
    def __init__(self):
        self.normal=0.;self.vertical=0.;self.rod='rung_06';self.original_calls=0
        self.data=SimpleNamespace(time=0.,contact=[SimpleNamespace(geom1=0,geom2=1)])
        environment=SimpleNamespace(data=self.data,model=SimpleNamespace(geom_bodyid=np.array([1,0])),
            robot_bodies={1},contact_groups={1:'left_hook'},contact_metrics=self.measurements)
        self.base=SimpleNamespace(env=environment,ends=np.array([113.66,227.32]),
            baseline=SimpleNamespace(command=self.original),last_phase='load_left_hook')
        self.state=install_hook_admission(self.base)

    def measurements(self):
        return {'left_hook_vertical_n':self.vertical,
                'contacts':[dict(group='left_hook',force_on_robot_world_n=[0,0,self.vertical],
                                 external_geom=self.rod)]}

    def contact_force(self,model,data,index,wrench):
        wrench[0]=self.normal

    @staticmethod
    def fake_qp(controller,qref,vref,active,caps):
        return np.zeros(29),dict(force_plan=np.tile([0.,0.,1.],2*sum(active)).tolist(),
                                base_equilibrium_residual=0.)

    def original(self,qref,vref,active):
        self.original_calls+=1
        return self.fake_qp(None,qref,vref,active,np.full(4,350.))

    def step(self,time,active=True,force=None,rod=None):
        self.data.time=time
        if force is not None:self.normal=self.vertical=force
        if rod is not None:self.rod=rod
        with patch('training.probe_hook_loading.mujoco.mj_contactForce',self.contact_force), \
             patch('training.probe_hook_loading.capped_command',self.fake_qp):
            return self.base.baseline.command(None,None,[True,True,active,True])


class ContactConfirmationTests(unittest.TestCase):
    def test_first_cycle_uses_original_even_when_contact_changes(self):
        fixture=ProbeFixture()
        fixture.step(10,False,0);fixture.step(20,True,0);fixture.step(21,True,100)
        self.assertEqual(fixture.original_calls,3)
        self.assertEqual(fixture.state['trace'],[])
        self.assertEqual(fixture.state['hooks'],{})

    def test_dropout_restarts_required_continuous_window(self):
        fixture=ProbeFixture();fixture.step(140.655,False,0)
        fixture.step(140.660,True,0)
        for tick in range(1,18):
            # A single missing sample at 30 ms invalidates the earlier 25 ms.
            fixture.step(140.660+.005*tick,True,0 if tick==6 else 10)
            if tick<17:self.assertIsNone(fixture.state['hooks'][2]['confirmed'])
        self.assertAlmostEqual(fixture.state['hooks'][2]['confirmed'],140.745)
        self.assertAlmostEqual(fixture.state['trace'][-1]['caps_per_contact_n'][2],7.5)
        fixture.step(140.995,True,10)
        self.assertAlmostEqual(fixture.state['trace'][-1]['caps_per_contact_n'][2],178.75)
        before=fixture.original_calls;fixture.step(141.245,True,10)
        self.assertEqual(fixture.original_calls,before+1)

    def test_wrong_rod_never_confirms_and_times_out(self):
        fixture=ProbeFixture();fixture.step(140.655,False,0)
        for tick in range(301):
            fixture.step(140.660+.005*tick,True,100,rod='rung_05')
        self.assertIsNone(fixture.state['hooks'][2]['confirmed'])
        with self.assertRaisesRegex(RuntimeError,'not confirmed within probe timeout'):
            fixture.step(142.165,True,100,rod='rung_05')

    def test_release_discards_previous_admission(self):
        fixture=ProbeFixture();fixture.step(140.655,False,0)
        for tick in range(12):fixture.step(140.660+.005*tick,True,10)
        self.assertIsNotNone(fixture.state['hooks'][2]['confirmed'])
        fixture.step(141.,False,0)
        self.assertNotIn(2,fixture.state['hooks'])
        fixture.step(141.005,True,0)
        self.assertIsNone(fixture.state['hooks'][2]['confirmed'])
        self.assertEqual(fixture.state['trace'][-1]['caps_per_contact_n'][2],7.5)


if __name__=='__main__':unittest.main()
