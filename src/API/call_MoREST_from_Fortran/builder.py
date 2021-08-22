import cffi
ffibuilder = cffi.FFI()

header = """
extern void call_morest_bias_sampling(int64_t *, double *, double *, int64_t *, double *, int64_t *, double *, int64_t *);
"""

with open('API_MoREST.py') as f:
    module = f.read()

with open("api_morest.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("api_morest", r'''
    #include "api_morest.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="../../../lib/libmorest.so", verbose=True) # for linux

