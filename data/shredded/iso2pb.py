#! /usr/bin/env python3

import sys, os
import numpy

sys.path.append('/fe3/demmapping/scripts')
import pbio

ii = 1
outdir = None

while ii < len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-o':
        outdir = sys.argv[ii]; ii += 1

for infname in sys.argv[ii:]:

    pts = []
    isovals = []
    with open(infname, 'rb') as inf:
        isoval = 0

        stem = os.path.basename( infname ).replace('_',' ')
        if 'q' in stem:
            ss = stem.split('q')
            isoval = - float( '0.' + ss[1].split()[0] )
        elif 'v' in stem:
            ss = stem.split('v')
            isoval = float( ss[1].split()[0] )

        for line in inf.readlines():
            ss = line.split()
            if len(ss) == 4 and ss[0] == b'v':
                pt = tuple(float(s) for s in ss[1:4])
                pts.append(pt)
                isovals.append( isoval )

    if outdir is None:
        outfname = infname.replace('.obj','') + '.pb'
    else:
        outfname = os.path.join( outdir, os.path.basename(infname).replace('.obj','') + '.pb' )
    pb = pbio.PbWriter( outfname, ['isoval'] )
    pb.writepcles( numpy.array( pts ), numpy.array(isovals).reshape(-1,1) )
    pb.close()
    print("Wrote %d points, isoval %g, to %s" % (len(pts), isoval, outfname))
