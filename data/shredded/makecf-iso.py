#! /usr/bin/env python3

import sys, os
import glob
import re

nnnnpat = re.compile('\.(\d\d\d\d)\.obj')

withbh = False
isovals = [ "q00005_7", "q0004_7", "q002_6", "q01_6" ]
isocolors = [ 3,           7,        11,       19    ]
stem = "isobh/m1b2plt50"

nnnns = []
for gotf in glob.glob( "%s%s*.obj" % (stem, isovals[0]) ):
    m = nnnnpat.search(gotf)
    if m is None:
        print("# funny filename ", gotf)
    else:
        nnnns.append( m.group(1) )
nnnns.sort()


print("#! /usr/bin/env partiview\n")

print("object g1=time")
print("tfm camera -.4 -.15 -1.5  0 0 0")
print("eval laxes off")
print("eval lsize 0.1")

for itime, nnnn in enumerate(nnnns):
    print('datatime %d' % itime)
    print('0 0 0 text %s' % nnnn)
    
 
for i, isoval in enumerate(isovals):
    isocolor = isocolors[i % len(isocolors)]
    print("object g%d=iso%s" % (i+2, isoval))
    print("")
    for itime, nnnn in enumerate(nnnns):
        print("waveobj -c %d -time %d %s%s.%s.obj" % (isocolor, itime, stem, isoval, nnnn))
    print("")

print("eval alpha 0.1") # on last (outermost) isossurface
