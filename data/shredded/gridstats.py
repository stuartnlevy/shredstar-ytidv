#! /gd/home/src/anaconda3/envs/py38/bin/python3

import sys, os
import re
import numpy
import pyopenvdb


gridname = 'density'

outstem = None
quantiles = [1, .999, .9, .5, 0]
vrange = None

ii = 1
while ii < len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-o':
        outstem = sys.argv[ii]; ii += 1
    elif opt == '-q':
        quantiles = [ float(s) for s in sys.argv[ii].replace(',',' ').split() ]
    elif opt == '-vrange':
        vrange = [float(s) for s in sys.argv[ii].replace(',',' ').split()]; ii += 1

if ii >= len(sys.argv):
    print("""Usage: %s [-q quantiles] [-o outstem] amrvdb5/*level9.??00.vdb
With -o, writes <outstem>.dat and <outstem>.gnuplot
With -q, replaces (comma-separated) quantiles list.  Default -q \"1,.999,.9,.5,0\"""" % sys.argv[0])
    sys.exit(1)

stuff = []

fmt = "  ".join( ["%11.7g"] * len(quantiles) ) + " # %s"

framepat = re.compile('\.(\d\d\d\d)\.vdb')

for vdbf in sys.argv[ii:]:
    gg, meta = pyopenvdb.readAll(vdbf)

    ggrid = [g for g in gg if g.name == gridname]
    if len(ggrid) != 1:
        print("Can't find unique '%s' grid in %s" % (gridname, vdbf))
        continue

    g = ggrid[0]
    
    gbox = g.evalActiveVoxelBoundingBox()

    agbox = numpy.array(gbox)

    if any(agbox[0] > agbox[1]):
        print("# Skipping empty grid from ", vdbf)
        continue

    arr = numpy.empty( (agbox[1] - agbox[0]) + 1 )
    g.copyToArray( arr, ijk=gbox[0] )

    akeep = arr[arr != 0]
    aquant = numpy.quantile( akeep, quantiles )
    print(fmt % (*aquant, vdbf))

    m = framepat.search(vdbf)
    if m is None:
        raise ValueError("Can't determine frame number from vdb file " + vdbf)

    
    frame = m.group(1)

    stuff.append( ( frame, aquant, vdbf ) )

if outstem:
    with open(outstem+'.dat', 'w') as datf:
        for frame, aquant, vdbf in stuff:
            print(frame, fmt % (*aquant, vdbf), file=datf)

    with open(outstem+'.gnuplot', 'w') as pltf:
        filetail = os.path.basename( stuff[0][2] )
        varname = filetail.split('_')[0]  # dens_ren_levelN.####.vdb => dens
        print('set title "quantiles of %s"' % varname, file=pltf)
        print('set logscale y', file=pltf)
        print('set style data linespoints', file=pltf)
        print('set terminal png size 960,720', file=pltf)
        print('set output "%s.png"' % outstem, file=pltf)
        print('print "Writing output to:  %s.png"' % outstem, file=pltf)
        replot = 'plot' if vrange is None else 'plot [:] [%g:%g]' % tuple(vrange)
        for i, q in enumerate(quantiles):
            sq = "max" if q==1 else "min" if q==0 else ("%g"%q)
            print('%s "%s.dat" using 1:%d title "%s %s"' % (replot, outstem, i+2, varname, sq), end='', file=pltf)
            replot = ','
        print('', file=pltf)

        print("Run: gnuplot %s.gnuplot  to produce  %s.png" % (outstem, outstem))
