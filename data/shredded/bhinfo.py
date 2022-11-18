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

        self.bhfname = None

        if datafilename.endswith('.dat'):
            self.bhfname = datafilename
            datafilename = os.path.dirname( datafilename )

        if os.path.isdir(datafilename):
            # if they just give a directory, choose the first suitable file
            for fi in glob.iglob( os.path.join( datafilename, 'multitidal_hdf5_plt_cnt_????' ) ):
                datafilename = fi
                break

        if self.bhfname is None:
            self.bhfname = os.path.join( os.path.dirname( datafilename ), "sinks_evol.dat" )

        self.sample_ds = yt.load(datafilename)

        self.inhale(self.bhfname)  # read position data 


    def inhale(self, sinksevolfile):

        # Given one data file's name, find the Black Hole etc. tracking info in the same directory.
        if not os.path.exists(sinksevolfile):
            raise OSError("BHInfo(): Can't find BH info file %s" % sinksevolfile)

        with open(sinksevolfile, 'rb') as inf:
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
                    if b'part_tag' in line:
                        self.headline = line.decode().rstrip()

        self.bhents = numpy.array(bhents)
        self.starents = numpy.array(starents)

    def extend_orbit_back(self, timefrom, timeto, extendout=None, nthrun=0, needtimes=None):
        """Fit a parabola to the (bh-star) positions over physical-time interval [timefrom, timeto].
           Overwrites the positions in self.bhents and self.starents before time=timefrom, with one entry for each time in needtimes[]."""

        # Let's say we don't need to find the plane.  We know it's in the XY plane, and with periapsis in the +Y direction.
        # Need to estimate angular momentum (equal area rule) to compute times along the curve.
        times = self.bhents[:,1]

        # selfcheck
        if any( times != self.starents[:,1] ):
            print("*** TROUBLE in sinks_evol: not all star times match BH times ***")


        chosen = (times >= timefrom) & (times <= timeto)  # subset

        iruns = numpy.where( times[1:] < times[:-1] )   
        runtrange = [times[0], times[-1]]
        if len(iruns) > 0 and len(iruns[0]) > 0:
            breaks = [0] + [k+1 for k in iruns[0]] + [len(times)]
            print("extend_orbit_back: sinks_evol contains multiple time runs:")
            for i in range(len(breaks)-1):
                trange = (times[breaks[i]], times[breaks[i+1]-1])
                if i == nthrun:
                    runtrange = trange
                print("%2d: [%4d:%4d] %g - %g" % (i, breaks[i],breaks[i+1], *trange))
            print("Using run %d (-nthrun N to change that)" % nthrun)
            chosen[0:breaks[nthrun]] = False
            if nthrun+1 < len(breaks):
                chosen[breaks[nthrun+1]:] = False

        if numpy.count_nonzero( chosen ) < 3:
            print("extend_orbit_back: fewer than 3 measurements in run %d's time range [%g .. %g] (have %d in range %g .. %g).  Can't fit a parabola." % (nthrun, timefrom, timeto, len(times), runtrange[0], runtrange[1]))
            return

        bsvec = self.bhents[:,2:5] - self.starents[:,2:5]
        ctimes = times[chosen]
        cbsvec = bsvec[chosen]

        # fit a parabola in Y.  Use the X coord, not time, as the independent variable.
        bsparabolay = Polynomial.fit( cbsvec[:,0], cbsvec[:,1], deg=2, domain=(cbsvec[0,0], cbsvec[-1,0]), window=[-1,1] )
        xbackward = cbsvec[0,0] - cbsvec[1,0] # positive (or negative) according to whether X increases (decreases) toward the extrapolated region

        y0 = bsparabolay(0)
        a = y0 - bsparabolay(1)

        # fit a line to the time curve assuming Kepler's equal-area law
        # indefinite integral of area with respect to x: r_vector dot perp(r'_vector) = (x,y) dot (-2ax^2, (y0 - ax^2)) = 
        # dA/dx = |(1,y',0) cross (x,y,0)| = |(0, 0, 1*y - y'*x)| = (y0-ax^2) - (-2ax^2) = y0 + ax^2
        # integral dA/dx = y0*x + 1/3 ax^3
        ##wrong## nomAs = bsvec[:,0] * (y0 + (1/3.0)*a*numpy.square(bsvec[:,0]))

        # debug: do direct estimate of area too
        # adopt a zero-point timestep for area
        xzeroes = numpy.where( cbsvec[1:,0] * cbsvec[:-1,0] < 0 )
        inear0 = xzeroes[0][0]+1 if (len(xzeroes) > 0 and len(xzeroes[0]) > 0) else 0
        crealdrvecs = cbsvec[1:] - cbsvec[:-1]
        crealdAs = []
        for rvec, drvec in zip(cbsvec[:-1], crealdrvecs):
            cx = numpy.cross( rvec, drvec )
            dA = numpy.sqrt( numpy.square(cx).sum() )
            crealdAs.append( dA )
        crealdAs.append( dA )
        crealA0 = numpy.sum( crealdAs[:inear0] )
        crealAs = numpy.cumsum( crealdAs ) - crealA0
        
        
        ##cnomAs = nomAs[chosen]
        ##crealAs = realAs[chosen]
        #time_A_fit = Polynomial.fit( cnomAs, ctimes, deg=1, domain=(cnomAs[0], cnomAs[-1]), window=[-1,1] )
        time_A_fit = Polynomial.fit( crealAs, ctimes, deg=1, domain=(crealAs[0], crealAs[-1]), window=[-1,1] )

        cdts = numpy.empty( len(ctimes) )
        cdts[:-1] = ctimes[1:] - ctimes[:-1]
        cdts[-1] = cdts[-2]

        # how good are the fits?
        if True:
            with open('/tmp/extend2.dat','w') as extf:
                print('#chosen x y t y_yfit t_tfit yfit tfit realA bhx bhy starx stary dt realdA', file=extf)
                for t, bsv, realA, bhpos, starpos, dt, realdA in zip(ctimes, cbsvec, crealAs, self.bhents[chosen,2:5], self.starents[chosen,2:5], cdts, crealdAs):
                    x, y = bsv[0:2]
                    yfit = bsparabolay( x )
                    #tfit = time_A_fit( nomA )
                    tfit = time_A_fit( realA )
                    print('1 %g %g %g\t%g %g\t%g %g\t%g %g %g %g %g %g %g' % (x,y,t, y-yfit, t-tfit, yfit,tfit, realA, bhpos[0],bhpos[1], starpos[0],starpos[1], dt, realdA), file=extf)

        if needtimes is not None:
            # how far to extend back?   Make a bunch of dA samples until time_A_fit(A) <= back_to_time
            curA = crealAs[0]
            curt = ctimes[0]
            curx = cbsvec[0,0]

            extA = [curA]
            extt = [curt]
            extx = [curx]
            exty = [bsparabolay(curx)]

            back_to_time = min( needtimes )

            xstep = 0.5 * xbackward

            prev_rvec = numpy.array( [curx, bsparabolay(curx), 0] )

            while extt[-1] > back_to_time:
                curx += xstep
                cury = bsparabolay(curx)
                rvec = numpy.array( [curx, cury, 0] )
                drvec = prev_rvec - rvec
                cx = numpy.cross( rvec, drvec )
                dA = numpy.sqrt( numpy.square(cx).sum() )
                curA -= dA
                extA.append( curA )
                extt.append( time_A_fit( curA ) )
                extx.append( curx )
                exty.append( cury )
                prev_rvec = rvec

            # Now find the x and y values at which time = needtimes
            # we reverse them so that numpy.interp() gets extt in ascending order
            
            exs = numpy.interp( needtimes, extt[::-1], extx[::-1] )
            eys = numpy.interp( needtimes, extt[::-1], exty[::-1] )

            iibase = numpy.where( chosen[1:] & ~chosen[:-1] )
            ibase = iibase[0][0]+1 if len(iibase[0]) > 0 else 0

            prebh = numpy.zeros( (len(needtimes), len(self.bhents[0])) )
            prestar = numpy.zeros( (len(needtimes), len(self.starents[0])) )

            prestar[:,0] = 65536
            prebh[:,0] = 131072
            prestar[:,2:5] = self.starents[ibase,2:5]
            prestar[:,1] = prebh[:,1] = needtimes
            prebh[:,2] = prestar[:,2] + exs
            prebh[:,3] = prestar[:,3] + eys

            # Now trash the original bhents[] and starents[], replacing the early portion with the newly extended stuff.
            # We don't fill in velocity or acceleration.
            # We assume star position is constant before the 'fit' interval

            obase = len(prebh)
            self.bhents = numpy.vstack( (prebh, self.bhents[ibase:]) )
            self.starents = numpy.vstack( (prestar, self.starents[ibase:]) )

            # remember those prefixes, for debugging
            self.prebh = prebh


            if '/' not in extendout:
                extendout = os.path.join( os.path.dirname(self.bhfname), extendout )
            print("# Writing to %s with range-extended BH and star positions (0..%d [%g..%g] new, %d..%d [%g..%g] old)" % (extendout, obase-1, self.bhents[0,1], self.bhents[obase-1,1], obase, len(self.bhents)-1, self.bhents[obase,1], self.bhents[-1,1]))
            with open(extendout, 'w') as expf:
                prevtime = 1e38
                fmt = " ".join(["%.14g" for i in range(len(self.bhents[0]))])
                for bhent, starent in zip(self.bhents, self.starents):
                    if bhent[1] < prevtime:
                        print(self.headline, file=expf)
                    print(fmt % tuple(bhent), file=expf)
                    print(fmt % tuple(starent), file=expf)
                    prevtime = bhent[1]

            return ibase

        return None

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
    force_recache = False

    nthrun = 0
    extendrange = (None, None)
    extendbase = 0.0
    extendstep = 5.0
    extendout = "sinks_evol.ext.dat"

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
        elif opt == '-recache':
            force_recache = True
        elif opt == '-nthrun':
            nthrun = int( sys.argv[ii] ); ii += 1
        elif opt == '-extend':
            extendrange = [ float(s) for s in sys.argv[ii].replace(',',' ').split() ]; ii += 1
        elif opt == '-extendout':
            extendout = sys.argv[ii]; ii += 1
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

    bhi.use_interval( *fitrange, redo=force_recache )

    if extendrange[0] is not None:
        # for all the timestep times preceding extendrange[0]
        ### pretimes = [steptime for steptime in bhi.times if steptime < extendrange[0]]
        pretimes = numpy.arange( extendbase, extendrange[0], extendstep )
        bhi.extend_orbit_back( *extendrange, extendout=extendout, nthrun=nthrun, needtimes=pretimes )
        sys.exit(0)
        # Recompute
        bhi.use_interval( *fitrange, redo=True )

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
        print("%4d  %11.7g  %13.10g %13.10g %13.10g   %13.10g %13.10g %13.10g  %11.7g  %g  %g %g %g  %g %g %g" % (step, bsdist, *bhpos, *starpos, bsspeed, *bhvel, *starvel, bhit['time']))

    if bhpath:
        with open(bhpath, 'w') as bhpf:
            for step in range(startstep, endstep+1):
                bhit = my_bhinfo_at_step(step)
                bhpos = bhi.mapped( bhit['pos'] )
                print("%13.10g %13.10g %13.10g 0 0 0 60 # %d of %d..%d BH position" % (*bhpos, step, *bhi.srange), file=bhpf)

    if trackpath:
        with open(trackpath, 'w') as trackf:
            for step in range(startstep, endstep+1):
                transla = ( bhi.mapped( bhi.translation_at( step ) ) + origin )
                print("%13.10g %13.10g %13.10g 0 0 0 60 # %d of %d..%d +translation" % (*transla, step, *bhi.srange), file=trackf)

    if invtrackpath:
        with open(invtrackpath, 'w') as trackf:
            for step in range(startstep, endstep+1):
                transla = - ( bhi.mapped( bhi.translation_at( step ) ) + origin )
                print("%13.10g %13.10g %13.10g 0 0 0 60 # %d of %d..%d -translation" % (*transla, step, *bhi.srange), file=trackf)

    if starpath:
        with open(starpath, 'w') as starpf:
            for step in range(startstep, endstep+1):
                bhit = my_bhinfo_at_step(step)
                starpos = bhi.mapped( bhit['starpos'] )
                print("%13.10g %13.10g %13.10g 0 0 0 60 # %d of %d..%d -translation" % (*starpos, step, *bhi.srange), file=starpf)
