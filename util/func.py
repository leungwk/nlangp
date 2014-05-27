import cPickle as pickle
import functools

# modified from http://stackoverflow.com/a/16464555 and https://wiki.python.org/moin/PythonDecoratorLibrary
def persist_to_file(file_name):
    def f_wrap(func):
        try:
            cache = pickle.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        @functools.wraps(func)
        def ret_f(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                pickle.dump(cache, open(file_name, 'w'))
            return cache[key]
        return ret_f

    return f_wrap
