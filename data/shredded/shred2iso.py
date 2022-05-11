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
threshes = []
outstem = "bden"
fieldvar = "density"
level = 5
outgrid = True
thresh = 3e-11
smoothed = False
fixbh = False
outcurve = None
doit = True
repair = False

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

    elif opt == '-thresh':
        arg = sys.argv[ii]; ii += 1
        for s in arg.split(','):
            ss = s.split('=')
            if len(ss) > 1:
                vstr = ss[1]
            else:
                vstr = ("v" + ss[0]).replace('v0.','v')   # 0.0025 => v0025
            threshes.append( [ float(ss[0]), vstr ] )

    elif opt == '-s' or opt == '-smoothed':
        smoothed = True
    elif opt == '-v':
        fieldvar = sys.argv[ii]; ii += 1
    elif opt == '-level' or opt == '-l':
        level = int( sys.argv[ii] ); ii += 1
    elif opt == '-ogrid':
        outgrid = True
    elif opt == '-outcurve':
        outcurve = sys.argv[ii]; ii += 1
    elif opt.startswith('-fixbh'):
        fixbh = True
        if '=' in opt:
            fixbh = [float(t) for t in opt.split('=')[1].split(',')]
    elif opt == '-n':
        doit = False
    elif opt == '-r':
        repair = True
    else:
        ii = len(sys.argv)

if ii >= len(sys.argv):
    print("""Usage: %s [-o outstem] [-q quantile=qlabel,quantile2=qlabel2,...] [-fixbh[=s0,s1]] [-l level] m1.0_p16_b2.0_300k_plt50/multitidal_hdf5_plt_cnt_0200 ...
Convert shredded-star gas density to isosurface(s) in .obj form.
 -o outstem  (-o bden) writes to <outstem>.NNNN.vdb
 -l amrgridlevel (-l 5  which yields 256^3 grid)
 -v fieldvar  (-v density)
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

        self.stepmap = None

        self.sample_ds = yt.load(datafilename)

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


    _interval_fields = ['srange', 'times', 'steps', 'pbh', 'pstar', 'ptranslate', 'pbh_S0offset', 'info_version']
    info_version = '0.8';

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

        if 'info_version' not in cache or cache['info_version'] != self.info_version:
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

    def _measure_interval(self, S0, S1, stepspacing=1):

        # Calibrate timestep->time mapping.   This involves opening lots of yt datasets.
        yt.set_log_level('warning')

        filelist = sorted( glob.glob( os.path.join( os.path.dirname(self.bhfname), "multitidal_hdf5_plt_cnt_????" ) ) )
        allfiles = []
        allsteps = []
        for fi in filelist:
            m = framepat.search(fi)
            if m is None:
                continue
            step = int( m.group(1) )
            if stepspacing <= 1 or len(allsteps)==0 or step - allsteps[-1] >= stepspacing:
                allsteps.append( step )
                allfiles.append( fi )

        if stepspacing > 1:
            ss = []

        i0, i1 = numpy.searchsorted( allsteps, [S0, S1] )
        i0 = max( 0, i0-1 )
        i1 = min( len(allsteps)-1, i1+1 )
            
        samples = {}
        steps = []
        stimes = []
        sbhposs = []
        sstarposs = []
        for fi in allfiles[0:i1+1]:
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
        self.sdomain = ( min(S0,steps[0]), S1 )
        self.times = stimes
        self.steps = steps

        myPoly = Chebyshev
        polydegree = min( max(3, len(steps)//3 + 1), len(steps)-1 )
        self.pbh = [ myPoly.fit( steps, bhposs[:,i], deg=polydegree, domain=self.sdomain, window=[-1,1] ) for i in range(3) ]
        self.pstar = [ myPoly.fit( steps, starposs[:,i], deg=polydegree, domain=self.sdomain, window=[-1,1] ) for i in range(3) ]


        # solve
        # T'(s) = lerp( frac(s, S0, S1), TBH'(s), TStar'(s) )
        # T(S1) = 0
        # T(s) = T(S1) + integral[S1,s]( T'(s) ) = T(S1) 

        
        # all the p*'s are functions of datastep-number s
        psx = Polynomial([-S0/(S1-S0), 1/(S1-S0)], domain=[0,1], window=[0,1]) # linear map S0->0, S1->1, linear in between
        psxS = psx.convert( kind=myPoly, domain=self.sdomain, window=[-1,1] )  # same linear map, converted to same domain as pbh and pstar
        plerpS = psxS**2 * (3 - 2*psxS)  # lerp(t,0,1) = 3t^2 - 2t^3 :: plerp(s) : S0->0, S1->1, cubic smoothstep in between
        ptprimeS = [ (1-plerpS)*(self.pbh[i].deriv()) + plerpS*(self.pstar[i].deriv()) for i in range(3) ]  # rate of change of translation: blending smoothly from pbh' to pstar'
        self.ptranslate = [ ptprimeS[i].integ(m=1, lbnd=S1) for i in range(3) ]  # ptranslate(s) = definite integral (from S1 to s) of ptprimeS.  Translation is 0 at s=S1.

        # calibrate what we'll use before and after the [S0,S1] window
        # before: stick with integral of BH position  (so we've extended the poly fit to include data as early as provided)
        # after: zero velocity, same translation as at s=S1

        # What's the difference between our ptranslate at s=S0 vs black hole interpolated position pbh at same time?
        self.pbh_S0offset = numpy.array([ self.ptranslate[i](S0) - self.pbh[i](S0) for i in range(3) ])

        # check fit
        for ppoly, pname, poss in [ (self.pbh,'pbh',bhposs), (self.pstar,'pstar',starposs) ]:
            print("# Worst fits to %s:" % pname, end="")
            for i in range(3):
                aa = numpy.array( [(ppoly[i](step) - poss[k,i]) for (k, step) in enumerate(steps)] )
                imin, imax = numpy.argmin(aa), numpy.argmax(aa)
                print(" {%g,%g}" % (aa[imin], aa[imax]), end="")
            print("")


    def use_interval(self, S0, S1, stepspacing=1, redo=False):

        if redo or not self._load_interval_info(S0,S1):
            self._measure_interval(S0,S1,stepspacing=stepspacing)
            self._save_interval_info()

    def mapped(self, p):
        # map domain box to 0..1
        return (numpy.array(p) - self.sample_ds.domain_left_edge.d) / (self.sample_ds.domain_right_edge.d - self.sample_ds.domain_left_edge.d)

    def translation_at(self, step):
        if self.srange is None:
            return numpy.zeros(3)

        S0, S1 = self.srange

        if step <= S0:
            # Before S0: track Black Hole using our interpolation curve, with constant offset for continuity after s=S0
            tr = [ self.pbh[i](step) + self.pbh_S0offset[i] for i in range(3) ]

        elif step >= S1:
            # After S1: since the star is stationary in sim coords, we track it by just keeping translation constant after s=S1
            tr = [ self.ptranslate[i](S1) for i in range(3) ]

        else:
            # use our interpolant between the two interpolating polynomials
            tr = [ self.ptranslate[i](step) for i in range(3) ]

        return numpy.array( tr )

    def _build_stepmap(self):
        if not hasattr(self, 'steps'):
            raise ValueError("Can't bhinfo_at_step() without calling use_interval() first")

        # fit a line to available step-to-time data
        self.stepmap = Polynomial.fit( self.steps, self.times, domain=[self.steps[0], self.steps[-1]], window=[0,1], deg=1 )  # 
        

    def bhinfo_at_step(self, step, fromfile=None):
        # map from step to time
        if self.stepmap is None:
            self._build_stepmap()

        attime = self.stepmap(step)
        return self.bhinfo_at_time(attime, fromfile)

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

if quants != [] and threshes != []:
    raise ValueError("Can't specify both -q and -thresh")

if quants == [] and threshes == []:
    raise ValueError("Must specify one of -q and -thresh")
    #quants = [ (quantdefault, "q%g"%quantdefault) ]



def grid2iso(grid, outobjname, field=fieldvar, left_edge=[0,0,0], right_edge=[1,1,1], vmin=1e-12, translate=numpy.array([0,0,0])):
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
    step = int(framestr)

    if repair:
        whats = (quants if quants else threshes)
        haveall = all( [ os.path.exists( '%s%s_%d.%s.obj' % (outstem, wstr, level, framestr) ) for (wval,wstr) in whats ] )
        if haveall:
            return

    bhi = BHInfo(fname)

    if isinstance(fixbh, (list,tuple,numpy.ndarray)):
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
            brick = ds.smoothed_covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=[fieldvar], num_ghost_zones=1)
        else:
            brick = ds.covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims, fields=[fieldvar], num_ghost_zones=0)

        bden = brick[fieldvar]
        # btem = brick[('gas','temperature')]

        if quants != []:
            vthreshes = numpy.quantile( bden, 1.0-numpy.array([val for (val,qstr) in quants]) )
        else:
            vthreshes = [val for (val,vstr) in threshes]
        vmax = bden.max()
    else:
        # dummies
        if quants != []:
            vthreshes = [ 1-float(qstr[1:]) for (val,qstr) in quants ]
        else:
            vthreshes = threshes
        vmax = 1
        bden = None

    whats = (quants if quants else threshes)
    for ((wval, wstr), vthresh) in zip(whats, vthreshes):
        outname = '%s%s_%d.%s.obj' % (outstem, wstr, level, framestr)
     
        print("vmax %g, vthresh %s => %g, for %s %s" % (vmax, wstr, vthresh, outstem, framestr))

        if fixbh:
            if bh is None:
                raise ValueError("Can't fix black hole position -- no sinks_evol.dat data")

            old_translate = - bhi.mapped(bh['pos'])
            translate = - bhi.mapped( bhi.translation_at( step ) )
            def ggg(v):
                return "%g %g %g" % tuple(v)
            print(f"# frame {framestr} translate {ggg(translate)} was {ggg(old_translate)}")
        else:
            translate = numpy.array([0,0,0])

        if doit:
            grid2iso( bden, outname, vmin=vthresh, translate=translate )


    if bh is not None:

        outbhspeck = outname.replace('.obj','') + '.bh.speck'
        with open(outbhspeck,'w') as speckf:
            print("# time %g" % bh['time'], file=speckf)
            print("%.11g %.11g %.11g" % tuple( bhi.mapped(bh['pos']) + translate ), "%g" % bh['mdot'], file=speckf )
        

    del ds, bden


### main ###

bhdatafile = sys.argv[ii]

if outcurve:

    bhi = BHInfo(bhdatafile)

    if fixbh not in [False,True]:
        bhi.use_interval( *fixbh )
        
    if bhi._load_interval_info(None,None):
        with open(outcurve, 'w') as curvef:
            print("# -fixbh=%g,%g  %s" % (*bhi.srange, bhdatafile), file=curvef)
            for step in numpy.arange(bhi.steps[0], bhi.steps[-1]+1):
                t = bhi.mapped( bhi.translation_at( step ) )
                tbh = bhi.mapped( [ bhi.pbh[i](step) for i in range(3) ] )
                print("%04d\t%10g  %10g  %10g\t%10g %10g %10g" % (step, *t, *tbh), file=curvef)
        print("Wrote to %s translation curve steps %04d .. %04d , tracking BH to %g and star after %g, from %s" % (outcurve, bhi.steps[0],bhi.steps[-1], *bhi.srange, bhdatafile))

else:
    process_star( bhdatafile )


