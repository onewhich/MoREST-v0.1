# my_module.py
from my_plugin import ffi
import numpy as np

# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        #print('ctype : ', ctype )
        #print('dtype : ', dtype )
        ctype2dtype[ctype] = np.dtype(dtype)

#ctype2dtype['int'] = np.dtype('int')

# Floating point types
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')

#print(ctype2dtype)

def asarray(ffi, ptr, shape, **kwargs):
    length = np.prod(shape)
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)
    # print( T )
    # print( ffi.sizeof( T ) )

    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    a = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T])\
          .reshape(shape, **kwargs)
    return a

@ffi.def_extern()
def add_one(a_ptr, n_ptr):
    n_ptr_type = ffi.getctype(ffi.typeof(n_ptr).item)
    n_atom = np.frombuffer((ffi.buffer(n_ptr, 2*ffi.sizeof(n_ptr_type))),ctype2dtype[n_ptr_type])[1]
    a = asarray(ffi, a_ptr, shape=(n_atom,3))
    print(a)
    a[:] += 1
    print(a)
