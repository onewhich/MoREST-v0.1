import os,sys

def where():
    print(os.path.split(os.path.abspath(__file__))[0])

