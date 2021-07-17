import os,sys

def where():
    print(os.path.split(os.path.abspath(__file__))[0])

sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'zz'))
import zz

where()
zz.where()
