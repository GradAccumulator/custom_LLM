

class ActivationFunction:
    def __call__(self, *args, **kwargs):
        if not hasattr(self,'forward'):
            raise AttributeError(f"[{self.__class__.__name__}] forward() method is required.")
        return self.forward(*args, **kwargs)