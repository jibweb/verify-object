from typing import Any, Union

def _set_value_recursively(obj, name: Union[list, str], val: Any):
    if type(name) == list:
        if len(name) == 1:
            setattr(obj, name[0], val)
        else:
            _set_value_recursively(getattr(obj, name[0]), name[1:], val)
    elif type(name) == str:
        if type(val) == dict:
            for subkey, subval in val.items():
                _set_value_recursively(getattr(obj, name), subkey, subval)
        else:
            setattr(obj, name, val)
    else:
        raise Exception("Wrong type given as input, should be (list, Any) or (str, dict)")


class BaseConfig:
    def __init__(self) -> None:
        print("Called before dataclass")
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__annotations__:
            raise Exception(f"In {self.__class__.__name__}: "
                            f"creation of attribute {name} is forbidden")

        if not self._expected_type(name, value):
            raise Exception(f"In {self.__class__.__name__}: "
                            f"invalid type when setting parameter {name} "
                            f"({type(value)} vs {self.__annotations__[name]})")

        super().__setattr__(name, value)

    def _expected_type(self, name: str, value: Any):
        """ Return True if type is as expected, and False otherwise
        """
        if self.__annotations__[name].__class__.__module__ == 'typing':
            print("Warning: typing types are not checked when setting attributes", self.__annotations__[name])
            return True

        return type(value) == self.__annotations__[name]


    def from_dict(self, new_values: dict, flat: bool=False):
        for name, new_val in new_values.items():
            if flat:
                split_name = name.split(".")
                _set_value_recursively(self, split_name, new_val)


# def configuration(_cls=None, *, init=True, repr=True, eq=True, order=False,
#                   unsafe_hash=False, frozen=False):

#     def wrap(cls):
#         print(cls)
#         cls = type(cls.__class__.__name__, (BaseConfig,), {})
#         return dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen)

#     return wrap(_cls)