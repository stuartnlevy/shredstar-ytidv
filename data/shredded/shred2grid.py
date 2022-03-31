#! /gd/home/src/anaconda3/envs/py38/bin/python3

#!/usr/bin/env python3

import sys, os
import yt
import numpy
import re

import pyopenvdb

#sys.path.append('/fe3/demmapping/scripts')
#import pbio


ii = 1
quant = 0.01
outstem = "bden"
level = 5
outgrid = True
thresh = 3e-11
smoothed = False

while ii < len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-o':
        outstem = sys.argv[ii]; ii += 1
    elif opt == '-q':
        quant = float( sys.argv[ii] ); ii += 1
    elif opt == '-s' or opt == '-smoothed':
        smoothed = True
    elif opt == '-level' or opt == '-l':
        level = int( sys.argv[ii] ); ii += 1
    elif opt == '-ogrid':
        outgrid = True
    else:
        ii = len(sys.argv)

if ii >= len(sys.argv):
    print("""Usage: %s [-o outstem] [-q quantile] m1.0_p16_b2.0_300k_plt50/multitidal_hdf5_plt_cnt_0200 ...
Convert shredded-star gas density to .vdb form, masked to preserve only the upper <quantile> fraction of the volume.
 -o outstem  (-o bden) writes to <outstem>.NNNN.vdb
 -l amrgridlevel (-l 5  which yields 256^3 grid)
 -q quant   (-q 0.01)""" % sys.argv[0], file=sys.stderr)
    sys.exit(1)

framepat = re.compile('_(\d\d\d\d)$')


def grid2vdb(grid, outvdbname, field='density', left_edge=[0,0,0], right_edge=[1,1,1], vmin=1e-12):
    sz, sy, sx = grid.shape

    dbox = numpy.array(right_edge) - numpy.array(left_edge)
    voxelsize = dbox / numpy.array( [sx, sy, sz] )

    grid[ grid <= vmin ] = 0.0

    vdbgrid = pyopenvdb.FloatGrid()

    vsz = voxelsize[0]
    vdbgrid.transform = pyopenvdb.createLinearTransform([
        [0, 0, vsz, 0],
        [0, vsz, 0, 0],
        [vsz, 0, 0, 0],  # is swapping X<->Z correct?   Or do we need a rotation?
        [0,  0,  0, 1] ])

    ijkout = tuple( numpy.rint( left_edge / voxelsize ).astype(numpy.int32) )

    vdbgrid.copyFromArray( grid.d, ijk=ijkout )
    
    metadata = dict(
        cmd=" ".join(sys.argv),
        var="density",
        log=str(False),
        scale="1",
        offset="0",
        thresh=thresh,
        smooth=str(smoothed),
        left="%.16g,%.16g,%.16g" % tuple(left_edge),
        right="%.16g,%.16g,%.16g" % tuple(right_edge),
        width="%.16g,%.16g,%.16g" % tuple(dbox),
        voxelsize="1/%.16g,%.16g,%.16g" % tuple( 1 / voxelsize ),
        level=str(level)
      )

    pyopenvdb.write( outvdbname, grids=[ vdbgrid ], metadata=metadata )
    svmin = "%g"%vmin
    
    print(f"Wrote {vdbgrid.activeVoxelCount()} nonempty cells to {outvdbname} with {field} >= {svmin} at {ijkout} of {sx,sy,sz}")


def process_star(fname):
    m = framepat.search( fname )
    if m is None:
        raise("Can't determine frame number from star data file " + fname)

    framestr = m.group(1)

    ds = yt.load( fname )
    ## ad = ds.all_data()

    dims = [ 8 << level ] * 3  # it's a FLASH file, with 8x8x8 blocks, so covering grid size is 8 * 2^level
    if smoothed:
        brick = ds.smoothed_covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=['density'], num_ghost_zones=0)
    else:
        brick = ds.covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=['density'], num_ghost_zones=0)

    bden = brick[('gas','density')]
    # btem = brick[('gas','temperature')]

    [ vthresh, vmax ] = numpy.quantile( bden, [1-quant, 1] )
    outname = '%s_%d.%s.vdb' % (outstem, level, framestr)
 
    grid2vdb( bden, outname, vmin=vthresh )

    del ds, bden


### main ###

process_star( sys.argv[ii] )
