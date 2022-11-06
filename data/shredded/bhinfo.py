#! /usr/bin/env python3

import sys, os
import glob
import re
import numpy
import pickle

from numpy.polynomial import Chebyshev, Polynomial, Hermite

import yt

framepat = re.compile('_(\d\d\d\d)$')


class BHInfo(object):
    def __init__(self, datafilename):

        self.bhents = None
        self.starents = None
        self.srange = (None, None)

        self.stepmap = None

        if os.path.isdir(datafilename):
            # if they just give a directory, choose the first suitable file
            for fi in glob.iglob( os.path.join( datafilename, 'multitidal_hdf5_plt_cnt_????' ) ):
                datafilename = fi
                break

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

        try:
            with open(cachef, 'rb') as picklef:
                cache = pickle.load(picklef)
        except Exception as e:
            print("bhinfo: couldn't load cache %s, will regenerate it: " % cachef, e)
            return False

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
            raise ValueError("Must call use_interval() before bhinfo_at_step()")

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
        ##print("# sought time %g found %g (delta = %g, tick %g)%s" % (time, dd['time'], dd['time']-time, tick, whence))

        return dd

        
    def bhinfo_at_tick(self, tick):

        nents = min( len(self.bhents), len(self.starents) )
        ii = min( int(tick), nents-1 )
        bhent = self.bhents[ii]
        starent = self.starents[ii]
        if tick != int(tick) and tick < nents-1:
            bhent = bhent + (tick - int(tick))*(self.bhents[ii+1] - bhent)  # bhent + ... rather than += ... -- be sure not to alter bhents[] array.
            starent = starent + (tick - int(tick))*(self.starents[ii+1] - starent)
        dd = dict(time=bhent[1], pos=bhent[2:5], vel=bhent[5:8], acc=bhent[8:11], mdot=bhent[15], starpos=starent[2:5], starvel=starent[5:8], staracc=starent[8:11])
        return dd


if __name__ == '__main__':

    def Usage():
        print("""Usage: %s [-bhpath outfile.wf] [-trackpath outtrack.wf] [-invtrackpath outinvtrack.wf] [-trackrange fromstep-tostep] simdir [fromstep-tostep] > motions.dat

""" % sys.argv[0])
        sys.exit(1)

    def frange_from(frange_s):
        ffrom, fto = (None, None) if frange_s is None else [ int(s) for s in frange_s.split('-') ] 
        return ffrom, fto

    bhpath = None
    starpath = None
    trackpath = None
    invtrackpath = None
    clampbeforestep = None
    outrange = (None, None)
    fitrange = (None, None)
    origin = numpy.array( [0, 0, 0] )

    ii = 1

    while ii < len(sys.argv) and sys.argv[ii][0] == '-':
        opt = sys.argv[ii]; ii += 1
        if opt == '-bhpath':
            bhpath = sys.argv[ii]; ii += 1
        elif opt == '-trackpath':
            trackpath = sys.argv[ii]; ii += 1
        elif opt == '-invtrackpath':
            invtrackpath = sys.argv[ii]; ii += 1
        elif opt == '-starpath':
            starpath = sys.argv[ii]; ii += 1
        elif opt == '-outrange':
            outrange = frange_from( sys.argv[ii] ); ii += 1
        elif opt == '-fitrange':
            fitrange = frange_from( sys.argv[ii] ); ii += 1
        elif opt == '-clampbefore':
            clampbeforestep = int( sys.argv[ii] ); ii += 1
        elif opt == '-origin':
            ss = sys.argv[ii].replace(',', ' ').split(); ii += 1
            if len(ss) != 3:
                raise ValueError("-origin: expected X,Y,Z -- what's " + sys.argv[ii-1] + " ?")
            origin = numpy.array( [ float(s) for s in ss ] )
        else:
            if opt != '-h':
                print("Unknown option: ", opt)
            Usage()
    
    if ii >= len(sys.argv):
        Usage()

    simdir = sys.argv[ii]

    bhi = BHInfo( simdir )

    bhi.use_interval( *fitrange )

    def my_bhinfo_at_step(step):
        mystep = step
        if clampbeforestep is not None and step < clampbeforestep:
            mystep = clampbeforestep

        return bhi.bhinfo_at_step( mystep )

    startstep, endstep = bhi.steps[0], bhi.steps[-1]
    if outrange[0] is not None:
        startstep, endstep = min(startstep, outrange[0]), max(endstep, outrange[1])

    print("#step   distance     bhX          bhY             bhZ       starX           starY       starZ        relspeed time  bhvelX  bhvelY  bhvelZ  starvelX starvelY starvelZ # %s %s" % (simdir, bhi.srange))
    for step in range(startstep, endstep+1):
        bhit = my_bhinfo_at_step(step)
        bhpos = bhi.mapped( bhit['pos'] )
        starpos = bhi.mapped( bhit['starpos'] )
        bhvel = bhi.mapped( bhit['vel'] )
        starvel = bhi.mapped( bhit['starvel'] )
        bsdist = numpy.sqrt( numpy.sum( numpy.square(bhpos-starpos) ) )
        bsspeed = numpy.sqrt( numpy.sum( numpy.square(bhvel-starvel) ) )
        print("%4d  %11.7g  %11.8g %11.8g %11.8g   %11.8g %11.8g %11.8g  %11.7g  %g  %g %g %g  %g %g %g" % (step, bsdist, *bhpos, *starpos, bsspeed, *bhvel, *starvel, bhit['time']))

    if bhpath:
        with open(bhpath, 'w') as bhpf:
            for step in range(startstep, endstep+1):
                bhit = my_bhinfo_at_step(step)
                bhpos = bhi.mapped( bhit['pos'] )
                print("%11.8g %11.8g %11.8g 0 0 0 60 # %d of %d..%d BH position" % (*bhpos, step, *bhi.srange), file=bhpf)

    if trackpath:
        with open(trackpath, 'w') as trackf:
            for step in range(startstep, endstep+1):
                transla = ( bhi.mapped( bhi.translation_at( step ) ) + origin )
                print("%11.8g %11.8g %11.8g 0 0 0 60 # %d of %d..%d +translation" % (*transla, step, *bhi.srange), file=trackf)

    if invtrackpath:
        with open(invtrackpath, 'w') as trackf:
            for step in range(startstep, endstep+1):
                transla = - ( bhi.mapped( bhi.translation_at( step ) ) + origin )
                print("%11.8g %11.8g %11.8g 0 0 0 60 # %d of %d..%d -translation" % (*transla, step, *bhi.srange), file=trackf)

    if starpath:
        with open(starpath, 'w') as starpf:
            for step in range(startstep, endstep+1):
                bhit = my_bhinfo_at_step(step)
                starpos = bhi.mapped( bhit['starpos'] )
                print("%11.8g %11.8g %11.8g 0 0 0 60 # %d of %d..%d -translation" % (*starpos, step, *bhi.srange), file=starpf)
