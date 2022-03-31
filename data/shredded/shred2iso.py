#! /gd/home/src/anaconda3/envs/py38/bin/python3

#!/usr/bin/env python3

import sys, os
import yt
import numpy
import re
import skimage.measure

#sys.path.append('/fe3/demmapping/scripts')
#import pbio


ii = 1
quantdefault = 0.01
quants = []
outstem = "bden"
level = 5
outgrid = True
thresh = 3e-11
smoothed = False
fixbh = False
doit = True

while ii < len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-o':
        outstem = sys.argv[ii]; ii += 1
    elif opt == '-q':
        arg = sys.argv[ii]; ii += 1
        for s in arg.split(','):
            ss = s.split('=')
            if len(ss) > 1:
                qstr = ss[1]
            else:
                qstr = ("q" + ss[0]).replace('q0.','q')   # 0.0025 => q0025
            quants.append( ( float(ss[0]), qstr ) )

    elif opt == '-s' or opt == '-smoothed':
        smoothed = True
    elif opt == '-level' or opt == '-l':
        level = int( sys.argv[ii] ); ii += 1
    elif opt == '-ogrid':
        outgrid = True
    elif opt == '-fixbh':
        fixbh = True
    elif opt == '-n':
        doit = False
    else:
        ii = len(sys.argv)

if ii >= len(sys.argv):
    print("""Usage: %s [-o outstem] [-q quantile=qlabel,quantile2=qlabel2,...] [-fixbh] [-l level] m1.0_p16_b2.0_300k_plt50/multitidal_hdf5_plt_cnt_0200 ...
Convert shredded-star gas density to isosurface(s) in .obj form.
 -o outstem  (-o bden) writes to <outstem>.NNNN.vdb
 -l amrgridlevel (-l 5  which yields 256^3 grid)
 -q quant   (-q 0.01)
 -fixbh     (if given, translate so that black hole is at (0,0,0))""" % sys.argv[0], file=sys.stderr)
    sys.exit(1)

framepat = re.compile('_(\d\d\d\d)$')


if quants == []:
    quants = [ (quantdefault, "q%g"%quantdefault) ]


def grid2iso(grid, outobjname, field='density', left_edge=[0,0,0], right_edge=[1,1,1], vmin=1e-12, translate=numpy.array([0,0,0])):
    sz, sy, sx = grid.shape

    dbox = numpy.array(right_edge) - numpy.array(left_edge)
    voxelsize = dbox / numpy.array( [sx, sy, sz] )

    ## grid[ grid <= vmin ] = 0.0

    vsz = voxelsize[0]

# marching_cubes(volume, level=None, *, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, method='lewiner', mask=None)

    verts, faces, _, _ = skimage.measure.marching_cubes(grid.d, level=vmin, spacing=voxelsize)

    import time
    t0 = time.time()

    with open(outobjname, 'wb') as outf:
        for v in verts:
            outf.write(b'v %.9g %.9g %.9g\n' % tuple(translate+v))
        for f in faces:
            outf.write(b'f %d %d %d\n' % tuple(f+1))
    dt = time.time() - t0
    print(f"Wrote {len(faces)} triangles cells to {outobjname} with {field} >= {vmin:g} in {int(dt*1000)} ms")


def bhinfo(fname, time):

    if hasattr(time, 'd'):
        time = time.d

    f = os.path.join( os.path.dirname(fname), "sinks_evol.dat" )
    if not os.path.exists(f):
        return None

    with open(f, 'rb') as inf:
        # [00]part_tag         [01]time         [02]posx         [03]posy         [04]posz         [05]velx         [06]vely         [07]velz       [08]accelx       [09]accely       [10]accelz        [11]anglx        [12]angly        [13]anglz         [14]mass         [15]mdot        [16]ptime
        bhents = []
        starents = []
        for line in inf.readlines():
            try:
                vv = [float(v) for v in line.split()]
                if vv[0] == 131072:
                    bhents.append( vv )
                elif vv[0] == 65536:
                    starents.append( vv )
            except:
                pass
    bhents = numpy.array(bhents)
    starents = numpy.array(starents)

    ii = numpy.searchsorted( bhents[:,1], time )
    bhent = bhents[ii]
    starent = starents[ii]
    print("# sought time %g found %g (delta = %g)" % (time, bhent[1], bhent[1]-time))
    return dict(time=bhent[1], pos=bhent[2:5], mdot=bhent[15], starpos=starent[2:5])


def mkdirfor(path):
    dirname, fname = os.path.split(path)
    if not os.path.isdir(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass


def process_star(fname):
    m = framepat.search( fname )
    if m is None:
        raise("Can't determine frame number from star data file " + fname)

    mkdirfor(outstem)

    framestr = m.group(1)

    ds = yt.load( fname )
    ## ad = ds.all_data()

    bh = bhinfo(fname, ds.current_time)

    # allow computing ghost zones in case smoothing is on, even though we may be lying about the data.
    if hasattr(ds, 'force_periodicity'):
        ds.force_periodicity()              # new way (yt >= 4.1?)
    else:
        ds.periodicity = (True, True, True) # old way.

    dims = [ 8 << level ] * 3  # it's a FLASH file, with 8x8x8 blocks, so covering grid size is 8 * 2^level on each axis

    if doit:
        if smoothed:
            brick = ds.smoothed_covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=['density'], num_ghost_zones=1)
        else:
            brick = ds.covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=['density'], num_ghost_zones=0)

        bden = brick['density']
        # btem = brick[('gas','temperature')]

        vthreshes = numpy.quantile( bden, 1.0-numpy.array([val for (val,qstr) in quants]) )
        vmax = bden.max()
    else:
        # dummies
        vthreshes = [ 1-float(qstr[1:]) for (val,qstr) in quants ]
        vmax = 1
        bden = None

    def mapped(p):
        # map domain box to 0..1
        return (numpy.array(p) - ds.domain_left_edge.d) / (ds.domain_right_edge.d - ds.domain_left_edge.d)

    for ((quant, qstr), vthresh) in zip(quants, vthreshes):
        outname = '%s%s_%d.%s.obj' % (outstem, qstr, level, framestr)
     
        print("vmax %g, vthresh %s => %g, for %s %s" % (vmax, qstr, vthresh, outstem, framestr))

        if fixbh:
            if bh is None:
                raise ValueError("Can't fix black hole position -- no sinks_evol.dat data")

            translate = - mapped(bh['pos'])
            def ggg(v):
                return "%g %g %g" % tuple(v)
            print(f"# frame {framestr} translate {ggg(translate)} = - bhpos {ggg(mapped(bh['pos']))} ( - bhpos {ggg(bh['pos']/1e13)})")
        else:
            translate = numpy.array([0,0,0])

        if doit:
            grid2iso( bden, outname, vmin=vthresh, translate=translate )


    if bh is not None:

        outbhspeck = outname.replace('.obj','') + '.bh.speck'
        with open(outbhspeck,'w') as speckf:
            print("# time %g" % bh['time'], file=speckf)
            print("%.11g %.11g %.11g" % tuple( mapped(bh['pos']) + translate ), "%g" % bh['mdot'], file=speckf )
        

    del ds, bden


### main ###

process_star( sys.argv[ii] )
