class OutputsContainer:
    __slots__ = ["_dict"]

    def __init__(self, **kwargs):
        super().__setattr__("_dict", dict(**kwargs))

    def __getattr__(self, item):
        return self._dict[item]

    def __getitem__(self, item):
        return self._dict[item]

    def __setattr__(self, key, item):
        raise TypeError("Cannot set attributes on OutputsContainer object")

    def __setitem__(self, key, item):
        self._dict[key] = item

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    # These are public methods but they are prefixed with _ to avoid name clashes with
    # actual output names
    def _values(self):
        return self._dict.values()

    def _items(self):
        return self._dict.items()

    def _get(self, item, dflt=None):
        return self._dict.get(item, dflt)

    def __repr__(self):
        return "OutputsContainer( %s )" % (",".join("%s=%r" % (k, v) for k, v in self._items()))
