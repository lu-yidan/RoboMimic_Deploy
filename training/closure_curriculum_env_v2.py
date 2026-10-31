"""Release solver/native-object reference cycles between suffix episodes."""
import gc
from training.closure_curriculum_env import ClosureCurriculumEnv


class ClosureCurriculumEnvV2(ClosureCurriculumEnv):
    def dispose_active(self):
        if self.env is not None:
            base = self.env.unwrapped
            for controller in (base.baseline, base.contact, base.contact.intervention):
                controller.cache.clear()
            self.env.close()
            self.env = None

    def reset(self, *, seed=None, options=None):
        self.dispose_active()
        gc.collect()
        return super().reset(seed=seed, options=options)

    def ensure_cached(self, case):
        if case not in self.caches:
            for env in self.caches.values():
                env.close()
            self.caches.clear()
            gc.collect()
        return super().ensure_cached(case)

    def close(self):
        self.dispose_active()
        super().close()
        self.caches.clear()
        gc.collect()
