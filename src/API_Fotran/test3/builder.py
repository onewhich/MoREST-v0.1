import cffi
ffibuilder = cffi.FFI()

header = """
extern void add_one (double *);
"""

with open('my_module.py') as f:
    module = f.read()

with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)
#ffibuilder.compile(target="libplugin.dylib", verbose=True) # for macos
ffibuilder.compile(target="libplugin.so", verbose=True) # for linux
