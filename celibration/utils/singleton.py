from weakref import WeakValueDictionary


class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances.keys():
            # This variable declaration is required to force a
            # strong reference on the instance.
            instance = super(Singleton, cls).__call__()
            init_func = getattr(instance, "initialize", None)
            if callable(init_func):
                init_func(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
