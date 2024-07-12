class EventHook(object):
    def __init__(self):
        self.__handlers = []  # [
        # h for h in self.__handlers if getattr(h, "im_self", False) != object
        # ]
        # This fix should be applied according to stackoverflow: https://stackoverflow.com/questions/1092531/event-system-in-python
        # But it doesnt work for us...:-)

    def __iadd__(self, handler):
        self.__handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.__handlers.remove(handler)
        return self

    def fire(self, *args, **kwargs):
        for handler in self.__handlers:
            handler(*args, **kwargs)

    def clearObjectHandlers(self, inObject):
        for theHandler in self.__handlers:
            if theHandler.im_self == inObject:
                self -= theHandler
