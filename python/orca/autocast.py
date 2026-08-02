from contextvars import ContextVar


_STATE = ContextVar("orca_autocast_state", default=(False, None))


class autocast:
    _enabled = False
    _dtype = None

    def __init__(self, device_type='cuda', dtype=None, enabled=True):
        self.device_type = device_type
        self.enabled = enabled
        import orca
        self.dtype = dtype or orca.DType.FLOAT16
        self._tokens = []

    def __enter__(self):
        state = (self.enabled, self.dtype if self.enabled else None)
        self._tokens.append(_STATE.set(state))
        autocast._enabled, autocast._dtype = state
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _STATE.reset(self._tokens.pop())
        autocast._enabled, autocast._dtype = _STATE.get()

    @classmethod
    def is_enabled(cls):
        return _STATE.get()[0]

    @classmethod
    def current_dtype(cls):
        return _STATE.get()[1]

class GradScaler:
    """
    Gradient scaling for mixed precision training.
    """
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        self._step_count = 0

    def scale_loss(self, loss):
        if not self.enabled:
            return loss
        return loss * self.scale
        
    def scale_tensor(self, tensor):
        # A wrapper for scale_loss in case people call scale(loss)
        return self.scale_loss(tensor)

    def step(self, optimizer):
        if not self.enabled:
            optimizer.step()
            return
            
        # 1. Unscale gradients
        inv_scale = 1.0 / self.scale
        for param in optimizer.parameters:
            grad = param.tensor.grad()
            if grad is not None:
                unscaled_grad = grad * inv_scale
                param.tensor.set_grad(unscaled_grad)
                
        # 2. Check for inf/nan in gradients
        found_inf = False
        for param in optimizer.parameters:
            grad = param.tensor.grad()
            if grad is not None:
                if grad.has_nan_or_inf():
                    found_inf = True
                    break
                    
        if not found_inf:
            optimizer.step()
        self._found_inf = found_inf

    def update(self):
        if not self.enabled:
            return
            
        if getattr(self, '_found_inf', False):
            self.scale *= self.backoff_factor
            self._step_count = 0
        else:
            self._step_count += 1
            if self._step_count >= self.growth_interval:
                self.scale *= self.growth_factor
                self._step_count = 0
