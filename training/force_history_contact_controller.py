"""Short force-confirmed contact persistence without a current geometric-gap query."""
from training.contact_memory_controller import ContactMemoryController


class ForceHistoryContactController(ContactMemoryController):
    def memory_points(self,key,body,t):
        if key not in self.support_memory:raise RuntimeError('No confirmed contact memory for '+key)
        old=self.support_memory[key];age=t-old['time_s']
        if age<0 or age>self.memory_window_s+1e-9:raise RuntimeError('Contact memory expired for '+key)
        rotation=self.d.xmat[body].reshape(3,3);origin=self.d.xpos[body]
        points=[(origin+rotation@local,frame.copy(),mu) for local,frame,mu,_,_ in old['points']]
        self.memory_events.append(dict(time_s=float(t),group=key,age_s=float(age),source='Historical force-confirmed point; no current contact or geometric gap claim',gap_query_used=False))
        return points
