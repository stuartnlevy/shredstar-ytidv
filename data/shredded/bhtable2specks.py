#! /usr/bin/env python3

import sys, os
import numpy

isoroot = ''
isostems = []
outstem = None
intable = None
clampbefore = None

cbh = 5
cstar = 24
radius = 0.005
nu, nv = 13, 9

def Usage():
    print("""Usage: %s [options...] from-to[%%incr]
Reads a something.bhtable.dat as written by bhinfo.py
Writes three .speck files:
   <outstem>.isos.speck  -- timed list of .pb files
   <outstem>.bh.speck -- timed track of Black Hole positions
   <outstem>.star.speck -- timed track of star positions
Options:
    -i infile.bhtable.dat
    -isoroot isobhb5_300/pb/b5_300
    -isos q00005_7,q0004_7,q002_6
    -cstar %d  -cbh %d  (color-index for star and black-hole)
    -clampbefore stepno
    -r radius
    -o outstem
    from-to[%%incr]
""" % (sys.argv[0], cstar, cbh))

    sys.exit(1)

ii = 1

while ii<len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-i':
        intable = sys.argv[ii]; ii += 1
    elif opt == '-o':
        outstem = sys.argv[ii]; ii += 1
    elif opt == '-isoroot':
        isoroot = sys.argv[ii]; ii += 1
    elif opt == '-isos':
        isos = sys.argv[ii]; ii += 1
        isostems.extend( isos.split(',') )
    elif opt.startswith('-clampbefore'):
        clampbefore = int( sys.argv[ii] ); ii += 1
    elif opt == '-r':
        radius = float( sys.argv[ii] ); ii += 1
    elif opt == '-cbh':
        cbh = int( sys.argv[ii] ); ii += 1
    elif opt == '-cstar':
        cstar = int( sys.argv[ii] ); ii += 1
    elif opt in ('-h', '--help'):
        Usage()
    else:
        print("Unknown option: ", opt)
        Usage()

if ii != len(sys.argv)-1:
    Usage()

if intable is None:
    print("Must specify -i infile.bhtable.dat")
    sys.exit(1)

if outstem is None:
    print("Must specify -o outstem")
    sys.exit(1)

frange = sys.argv[ii]

ss = frange.split('%')
fincr = int(ss[1]) if len(ss)>1 else 1
ss = ss[0].split('-')
ffrom, fto = int(ss[0]), int(ss[-1])

def mksphere(nu, nv, r=1, cen=[0,0,0]):
    pts = numpy.empty( (nv,nu,3) )
    for iv in range(nv):
        thv = (numpy.pi*iv)/(nv-1)
        vs, vc = numpy.sin(thv), numpy.cos(thv)
        for iu in range(nu):
            phiu = (2*numpy.pi*iu)/nu
            us, uc = numpy.sin(phiu), numpy.cos(phiu)
            pts[iv,iu,:] = vs*uc, vs*us, vc
    pts *= r
    pts += numpy.array(cen).reshape(1,1,3)
    return pts


bhpos = {}
starpos = {}

# #step   distance     bhX          bhY             bhZ       starX           starY       starZ        mdot   time # 1e3_m1.0_p16_b5.0_300k (498, 800)
#  0   0.00990263   0.50791666  0.49405853         0.5     0.4999944   0.4999999         0.5            0  11624.3
 
with open(intable) as inf:
    for line in inf.readlines():
        ss = line.split()
        if len(ss) <= 7 or ss[0][0] == '#':
            continue

        step = int(ss[0])
        if step==clampbefore or (step >= ffrom and step <= fto and (step - ffrom) % fincr == 0):
            bhpos[step] = [float(s) for s in ss[2:5]]
            starpos[step] = [float(s) for s in ss[5:8]]

isosfname = "%s.isos.speck" % outstem
bhfname = "%s.bh.speck" % outstem
starfname = "%s.star.speck" % outstem
print("Writing %s and %s and %s" % (isosfname, bhfname, starfname))

with open(bhfname, 'w') as outbhf, \
     open(starfname, 'w') as outstarf, \
     open(isosfname, 'w') as outisosf:

    for otime, frameno in enumerate( range(ffrom, fto+1, fincr) ):
        if clampbefore is not None:
            frameno = max(clampbefore, frameno)

        print("datatime %d # step %d" % (otime, frameno), file=outisosf)
        for isostem in isostems:
            print("pb -t %d %s%s.%04d.pb" % (otime, isoroot, isostem, frameno), file=outisosf)

        for posmap, tofile, cindex in (bhpos, outbhf, cbh), (starpos, outstarf, cstar):

            print("datatime %d # step %d" % (otime, frameno), file=tofile)
            print("0 0 0", file=tofile)
            if frameno in posmap:
                print("mesh -time %d -s wire -c %d {" % (otime, cindex), file=tofile)
                print("%d %d" % (nu,nv), file=tofile)
                pts = mksphere(nu, nv, r=radius, cen=posmap[frameno])
                for pt in pts.reshape(-1, 3):
                    print("%g %g %g" % tuple(pt), file=tofile)
                print("}", file=tofile)
            else:
                print("# No data for timestep %d" % frameno, file=tofile)
