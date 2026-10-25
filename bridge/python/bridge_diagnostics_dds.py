"""Optional read-only diagnostics; existing state/command DDS wire types stay unchanged."""
from dataclasses import dataclass, field
import numpy as np
import cyclonedds.idl as idl
import cyclonedds.idl.types as types
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from bridge.python.bridge_state_dds import _BRIDGE_QOS

@dataclass
class BridgeDiagnostics(idl.IdlStruct, typename="mjlab::msg::dds_::BridgeDiagnostics_"):
    tick: types.uint64 = 0
    tau_est: types.sequence[types.float32] = field(default_factory=list)
    ddq: types.sequence[types.float32] = field(default_factory=list)
    imu_accel: types.sequence[types.float32] = field(default_factory=list)
    motor_temperature: types.sequence[types.float32] = field(default_factory=list)

class DiagnosticsPublisher:
    def __init__(self, participant, state_topic):
        self.topic = Topic(participant, state_topic + "_diagnostics", BridgeDiagnostics, qos=_BRIDGE_QOS)
        self.writer = DataWriter(participant, self.topic, qos=_BRIDGE_QOS)

    def publish(self, low_state):
        motors = low_state.motor_state[:29]
        self.writer.write(BridgeDiagnostics(
            tick=int(low_state.tick), tau_est=[float(m.tau_est) for m in motors],
            ddq=[float(m.ddq) for m in motors],
            imu_accel=[float(v) for v in low_state.imu_state.accelerometer],
            motor_temperature=[float(v) for m in motors for v in m.temperature],
        ))

class DiagnosticsSubscriber:
    def __init__(self, participant, state_topic):
        self.topic = Topic(participant, state_topic + "_diagnostics", BridgeDiagnostics, qos=_BRIDGE_QOS)
        self.reader = DataReader(participant, self.topic, qos=_BRIDGE_QOS)
        self.cache = {}

    def matching(self, tick):
        for msg in self.reader.take(N=64):
            if isinstance(msg, BridgeDiagnostics):
                self.cache[int(msg.tick)] = msg
        while len(self.cache) > 64:
            del self.cache[next(iter(self.cache))]
        return self.cache.get(int(tick))

def apply_diagnostics(state, msg):
    """Never reuse stale torque; only an exactly matching LowState tick is supplied."""
    state.tau_est = np.full(29, np.nan, dtype=np.float32)
    state.tau_est_valid = False
    for key, n in (("imu_accel", 3), ("motor_ddq", 29), ("motor_temperature", 58)):
        state.telemetry[key] = np.full(n, np.nan, dtype=np.float32)
    if msg is None:
        return
    for field_name, key, n in (("ddq", "motor_ddq", 29), ("imu_accel", "imu_accel", 3),
                               ("motor_temperature", "motor_temperature", 58)):
        x = np.asarray(getattr(msg, field_name), dtype=np.float32)
        if x.shape == (n,): state.telemetry[key] = x
    tau = np.asarray(msg.tau_est, dtype=np.float32)
    if tau.shape == (29,) and np.isfinite(tau).all():
        state.tau_est = tau.copy()
        state.tau_est_valid = True
