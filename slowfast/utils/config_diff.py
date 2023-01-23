from itertools import zip_longest
from fvcore.common.config import CfgNode


_match=type('match', (object,), {})()
_missing=type('missing', (object,), {})()
_missing.__class__.__str__ = lambda self: '(nothing)'
fmt = lambda a, b: f'{a} -> {b}'


def _deep_match(a, b, fmt=fmt, inverse=False):
    """Get change in config"""
    if isinstance(a, (dict, CfgNode)) and isinstance(a, (dict, CfgNode)):
        o = {k: _deep_match(a[k], b[k], fmt, inverse) for k in b if k in a}
        o = {k: v for k, v in o.items() if v is not _match}
        if o:
            return o
    elif isinstance(a, (list, tuple, set)) and isinstance(b, (list, tuple, set)):
        o = [_deep_match(ai, bi, fmt, inverse) for ai, bi in zip_longest(a, b, fillvalue=_missing)]
        if all(x is not _match for x in o) if inverse else not all(x is _match for x in o):
            return [ai if x is _match else x for ai, x in zip_longest(a, o, fillvalue=_missing)]
    elif (a != b) != inverse:  # scalar, bool, str
        return fmt(a, b)
    return _match

def deep_match(a, b, fmt=fmt):
    x = _deep_match(a, b, fmt=fmt)
    return None if x is _match else x

def deep_mismatch(a, b):
    x = _deep_match(a, b, fmt=lambda a, b: a, inverse=True)
    return None if x is _match else x

def maybe_load_config(cfg):
    if isinstance(cfg, str):
        import yaml
        with open(cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg

def print_compare(a, b):
    import yaml
    a = maybe_load_config(a)
    b = maybe_load_config(b)

    print("-- Config Changes: -----")
    print(yaml.dump(deep_match(a, b)))
    print("-- Redundant Config: -----")
    print(yaml.dump(deep_mismatch(a, b)))

if __name__ == '__main__':
    import fire
    fire.Fire(print_compare)