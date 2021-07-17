import os,sys
sys.path.append('./zz')
import zz

def where():
    print(os.path.split(os.path.abspath(__file__))[0])


where()
zz.where()
