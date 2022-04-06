#! /gd/home/src/anaconda3/bin/python3

import sys, os
import yt
import numpy
import re
import skimage.measure

import pickle

import glob
from numpy.polynomial import Chebyshev, Polynomial, Hermite

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
    elif opt.startswith('-fixbh'):
        fixbh = True
        if '=' in opt:
            fixbh = [float(t) for t in opt.split('=')[1].split(',')]
    elif opt == '-n':
        doit = False
    else:
        ii = len(sys.argv)

if ii >= len(sys.argv):
    print("""Usage: %s [-o outstem] [-q quantile=qlabel,quantile2=qlabel2,...] [-fixbh[=s0,s1]] [-l level] m1.0_p16_b2.0_300k_plt50/multitidal_hdf5_plt_cnt_0200 ...
Convert shredded-star gas density to isosurface(s) in .obj form.
 -o outstem  (-o bden) writes to <outstem>.NNNN.vdb
 -l amrgridlevel (-l 5  which yields 256^3 grid)
 -q quant   (-q 0.01)
 -fixbh     (if given, translate so that black hole is at (0,0,0))
 -fixbh=s0,s1 --- smoothly interpolate from tracking BH to tracking star during datastep range s0 ... s1.
    Star ends up at 0,0,0 at times >= s1.
 """ % sys.argv[0], file=sys.stderr)
    sys.exit(1)

framepat = re.compile('_(\d\d\d\d)$')


class BHInfo(object):
    def __init__(self, datafilename):

        self.bhents = None
        self.starents = None

        self.inhale(datafilename)  # read position data 


    def inhale(self, datafilename):

        # Given one data file's name, find the Black Hole etc. tracking info in the same directory.
        self.bhfname = os.path.join( os.path.dirname(datafilename), "sinks_evol.dat" )
        if not os.path.exists(self.bhfname):
            raise OSError("BHInfo(): Can't find BH info file %s" % self.bhfname)

        with open(self.bhfname, 'rb') as inf:
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
        self.bhents = numpy.array(bhents)
        self.starents = numpy.array(starents)


    _interval_fields = ['srange', 'times', 'steps', 'pbh', 'pstar', 'ptranslate']

    def _load_interval_info(self, S0,S1):

        cachef = self.bhfname + ".cache.pickle"

        # there might be a cached file with a time curve etc. in it

        if not os.path.exists(cachef):
            return False

        with open(cachef, 'rb') as picklef:
            cache = pickle.load(picklef)

        if S0 is not None and tuple(cache['srange']) != (S0,S1):
            # Cache exists but spans wrong datastep range
            return False

        for field in self._interval_fields:
            # self.srange = cache['srange'] etc.
            # pbh, pstar, ptranslate are polynomial functions of datastep number
            setattr(self, field, cache[field])
    
        return True

    def _save_interval_info(self):
        cachef = self.bhfname + ".cache.pickle"

        dd = dict()
        for field in self._interval_fields:
            dd[field] = getattr(self, field)

        with open(cachef, 'wb') as picklef:
            pickle.dump( dd, picklef )

    def _measure_interval(self, S0, S1):

        # Calibrate timestep->time mapping.   This involves opening lots of yt datasets.
        yt.set_log_level('warning')

        allfiles = sorted( glob.glob( os.path.join( os.path.dirname(self.bhfname), "multitidal_hdf5_plt_cnt_????" ) ) )
        allsteps = []
        for fi in allfiles:
            m = framepat.search(fi)
            if m is None:
                continue
            allsteps.append( int( m.group(1) ) )

        i0, i1 = numpy.searchsorted( allsteps, [S0, S1] )
        i0 = max( 0, i0-1 )
        i1 = min( len(allsteps)-1, i1+1 )
            
        samples = {}
        steps = []
        stimes = []
        sbhposs = []
        sstarposs = []
        for fi in allfiles[i0:i1+1]:
            stime = None
            m = framepat.search(fi)
            if m is None:
                continue
            sstep = int( m.group(1) )

            try:
                sds = yt.load( fi )
                stime = float(sds.current_time.d)
                bhi = self.bhinfo_at_time(stime, fi)
                if abs(stime - bhi['time']) < 20:
                    steps.append(sstep)
                    stimes.append(bhi['time'])  # approximately equal to stime
                    sbhposs.append(bhi['pos'])
                    sstarposs.append(bhi['starpos'])
                del sds
            except OSError as e:
                print("# use_interval(): Not fitting troublesome timestep %d  (time %s) %s: %s" % (sstep, stime, fi, e))

        bhposs = numpy.array(sbhposs)
        starposs = numpy.array(sstarposs)


        yt.set_log_level('info')

        self.srange = (S0, S1)
        self.times = stimes
        self.steps = steps

        myPoly = Chebyshev
        polydegree = min( max(3, len(steps)//3 + 1), len(steps)-1 )
        self.pbh = [ myPoly.fit( steps, bhposs[:,i], deg=polydegree, domain=[S0,S1], window=[-1,1] ) for i in range(3) ]
        self.pstar = [ myPoly.fit( steps, starposs[:,i], deg=polydegree, domain=[S0,S1], window=[-1,1] ) for i in range(3) ]


        # solve
        # T'(s) = lerp( frac(s, S0, S1), TBH'(s), TStar'(s) )
        # T(S1) = 0
        # T(s) = T(S1) + integral[S1,s]( T'(s) ) = T(S1) 

        
        # all the p*'s are functions of datastep-number s
        psx = Polynomial([-S0/(S1-S0), 1/(S1-S0)], domain=[0,1], window=[0,1]) # linear map S0->0, S1->1, linear in between
        psxS = psx.convert( kind=myPoly, domain=[S0,S1], window=[-1,1] )  # same linear map, converted to same domain as pbh and pstar
        plerpS = psxS**2 * (3 - 2*psxS)  # lerp(t,0,1) = 3t^2 - 2t^3 :: plerp(s) : S0->0, S1->1, cubic smoothstep in between
        ptprimeS = [ plerpS*(self.pbh[i].deriv()) + (1-plerpS)*(self.pstar[i].deriv()) for i in range(3) ]  # rate of change of translation: blending smoothly from pbh' to pstar'
        self.ptranslate = [ ptprimeS[i].integ(m=1, lbnd=S1) for i in range(3) ]  # ptranslate(s) = definite integral (from S1 to s) of ptprimeS.  Translation is 0 at s=S1.

        # check fit
        for ppoly, pname, poss in [ (self.pbh,'pbh',bhposs), (self.pstar,'pstar',starposs) ]:
            print("# Worst fits to %s:" % pname, end="")
            for i in range(3):
                aa = numpy.array( [(ppoly[i](step) - poss[k,i]) for (k, step) in enumerate(steps)] )
                imin, imax = numpy.argmin(aa), numpy.argmax(aa)
                print(" {%g,%g}" % (aa[imin], aa[imax]), end="")
            print("")


    def use_interval(self, S0, S1):

        if not self._load_interval_info(S0,S1):
            self._measure_interval(S0,S1)
            self._save_interval_info()

    def translation_at(self, step):
        if self.srange is None:
            return numpy.zeros(3)

        if step <= self.srange[0]:
            pass

        elif step >= self.srange[1]:
            pass

        else:
            return numpy.array( [ self.ptranslate[i](step) for i in range(3) ] )

        
    def bhinfo_at_time(self, time, fromfile=None):

        if hasattr(time, 'd'):
            time = time.d

        tick = numpy.searchsorted( self.bhents[:,1], time )
        if tick < len(self.bhents)-1 and self.bhents[tick,1] < self.bhents[tick+1,1]:
            tick += (time - self.bhents[tick,1]) / (self.bhents[tick+1,1] - self.bhents[tick,1])
        dd = self.bhinfo_at_tick(tick)

        # debuggg
        whence = "" if fromfile is None else " for frame " + fromfile[-4:]
        print("# sought time %g found %g (delta = %g, tick %g)%s" % (time, dd['time'], dd['time']-time, tick, whence))

        return dd

        
    def bhinfo_at_tick(self, tick):

        nents = min( len(self.bhents), len(self.starents) )
        ii = min( int(tick), nents-1 )
        bhent = self.bhents[ii]
        starent = self.starents[ii]
        if tick != int(tick) and tick < nents-1:
            bhent = bhent + (tick - int(tick))*(self.bhents[ii+1] - bhent)  # bhent + ... rather than += ... -- be sure not to alter bhents[] array.
            starent = starent + (tick - int(tick))*(self.starents[ii+1] - starent)
        dd = dict(time=bhent[1], pos=bhent[2:5], mdot=bhent[15], starpos=starent[2:5])
        return dd


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

    bhi = BHInfo(fname)

    if isinstance(fixbh, (list,tuple)):
        bhi.use_interval( fixbh[0], fixbh[1] )

    ds = yt.load( fname )
    ## ad = ds.all_data()

    bh = bhi.bhinfo_at_time(ds.current_time)

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


