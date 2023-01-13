#! /usr/bin/env python3

import sys, os
import re
import numpy
import getopt


import sdb

sys.path.append('/fe3/demmapping/scripts')
import bgeo

mwaysdbfile = "/fe4/ghez2/sr/mwaysnapshot.sdb"
outstem = "../data/mway/"

mwaysdbpopfile = "/fe4/ghez2/data/mway/allparts.pop"  # sdbpop output from $RDATA/mway/GCparts selected pieces
##grp  count  radius         opacity          mag    rmin gmin bmin     r   g   b      rmax gmax bmax  types filename
# 66  191719 7.78e-05 0.00029<0.0012<0.00166  -1.312  0.10 0.00 0.00 < 0.39 0.00 0.00 < 0.55 0.00 0.00     B  B1B2cloud.sdb
# 16   71685 3.88e-05 0.00003<0.0013<0.00678  -1.293  0.16 0.05 0.00 < 0.34 0.10 0.03 < 0.42 0.21 0.06     B  NTFcloud.sdb


never = False  # -n option: test only, don't overwrite anything
always = False # -y option: overwrite files even if they already exist
verbose = False
withgroup = False

ignoregroups = set( [21, 22, 24] + [76, 85, 130,131,132] ) # UN_GASSY + UN_BH


opts, _ = getopt.getopt( sys.argv[1:], "nyvgo:i:" )
for opt, arg in opts:
    if opt == '-n':
        never = True
    elif opt == '-y':
        always = True
    elif opt == '-v':
        verbose = True
    elif opt == '-g':
        withgroup = True
    elif opt == '-o':
        outstem = arg
    elif opt == '-i':
        mwaysdbfile = arg


print(f"""{sys.argv[0]} parameters:
 -o {outstem}  (output directory)
 -i {mwaysdbfile}  (mway sdb file, saved from 'set sdb' in some .header)

mwaysdbpop file, output from sdbpop on assorted .sdb files, listing other groups which may be wanted:
    {mwaysdbpopfile}""", flush=True)

tfmcmd = "gettfm m83disk cgal_gc195"

# clump definitions from sample header file.   'grep "^set [^ ]*_color =" mwaywhatever.header'
clumpdefs = \
"""
set plume_color =  "sdbrecolor -g 53-57 -f .5 -O -c .35,.05,.02@.0001 -c .35,.08,.03@.001 -c .3,.14,.05@.002"
set B1B2cloud_color = "sdbrecolor -g 66 -f .75 -O -c .15,.06,.02@.0004 -c .2,.1,.05@.001 -c .3,.22,.1@.0015"
set sgrB1B2_color = "sdbrecolor -g 64-65 -f .25 -O -c .3,.1,.05@.0001 -c .33,.2,.09@.001 -c .4,.3,.13@.004"
set newSNR_color = "sdbrecolor -g 67 -f .75 -c .0,.12,.2" #3-8
set sgrCinner_color = "sdbrecolor -g 68 -f 1 -O -c .25,.05,.02@.0007 -c .2,.1,.05@.001 -c .2,.16,.04@.0015"
set sgrCouter_color = "sdbrecolor -g 69 -f 1 -O -c .04,.15,.10@.0007 -c .07,.1,.15@.001  -c .12,.18,.22@0014"
set coherent_color = "sdbrecolor -g 70 -f 1 -O -c .15,.06,.02@.0004 -c .2,.08,.04@.001 -c .25,.16,.08@.0015" #3-7
set backMolec_color = " sdbrecolor -g 71 -f .75 -c .25,.06,.01"
set centerCloud_color = "sdbrecolor -g 72 -f .25 -c .4,.042,.02 -C .36,.28,.075"
set sgrD_SNR_color = "sdbrecolor -g 61 -f .65 -c .03,.10,.15"
set sgrD_HII_color = "sdbrecolor -g 62 -f .6 -c .25,.14,.05 -C .3,.2,.07"
set SNR_0901_color = "sdbrecolor -g 63 -f .1 -c .15,.25,.3"
set sagA_therm_color = "sdbrecolor -g 82 -f 1 -n -O -c .35,.06,0@.0002 -c .32,.11,.02@.00037  -c .3,.20,.06@.00045" #2-8
set filaBG_color = "sdbrecolor -g 11 -O -f .75 -c .3,.07,.02@0004 -c .3,.25,.09@004"
set filaMid_color = "sdbrecolor -g 12 -f .6 -O -c .3,.08,.03@0003 -c .3,.25,.07@005" 
set filaCU_color = "sdbrecolor -g 15 -f 1 -O -c .3,.1,.05@.0001 -c .3,.2,.05@.002"
set NTFcloud_color = "sdbrecolor -g 16 -f .75 -O -c .25,.08,.02@.0004 -c .25,.12,.03@.001 -c .3,.18,.04@.005"
set sickle_color = "sdbrecolor -g 17 -f .5 -c .35,.06,.02@0002 -c .3,.2,.08@.001 -c .3,.24,.1@0018"
set scuba_color = "sdbrecolor -g 50 -f .5 -O -c .3,.15,.05@.0002 -c .39,.19,.1@.0023  -c .33,.06,.02@.01"
"""

clumppat = re.compile('set (\S+)_color\s*=\s*"sdbrecolor -g (\d[-\d]*)')


###

with os.popen(tfmcmd, 'r') as tfmf:
    sTcgal = tfmf.readline()
    Tcgal = numpy.fromstring(sTcgal, dtype=float, count=16, sep=' ').reshape( 4, 4 )
    sTcgalcommas = ",".join( sTcgal.split() )

def process_clump( mwsdb, wanted, outstemclumpname, groups ):
    global never, always, verbose

    def maybefile(fname):
        exists = os.path.exists(fname)
        if exists:
            msg = ("Would overwrite existing file" if never else "Overwriting existing file") if always else "Not overwriting existing file"
        else:
            msg = "Would write file" if never else "Writing file"
        if never:
            msg = "#dry run# " + msg
        doit = always or (not never) or (not exists)
        return doit, msg
        

    numpts = numpy.count_nonzero( wanted )
    if numpts == 0:
        print("## No data to write to %s.{sdb,bgeo} -- skipping." % outstemclumpname, flush=True)
        return
    
    if isinstance(groups, int):
        sdbsift = "sdbsift group == %d" % groups
    elif isinstance(groups, slice):
        sdbsift = "sdbsift group %d %d" % (groups.start, groups.stop+1)
    else:
        sdbsift = "sdbsift -o " + " ".join(["group == %d" % g for g in groups])

    outsdb = outstemclumpname + '.sdb'
    sdbcmd = ("%s < '%s' | sdbshift -t %s > '%s'" % (sdbsift, mwaysdbfile, sTcgalcommas, outsdb))
    if verbose:
        print(("##dry run##" if never else "#"), sdbcmd, flush=True)

    doit, msg = maybefile(outsdb)
    print("#", msg, outsdb, "(%d points)" % numpts, flush=True)
    if doit:
        os.system(sdbcmd)

    outfname = outstemclumpname + ".bgeo"

    pointattrnames = ['radius', 'opacity', 'Cr','Cg','Cb']
    if withgroup:
        pointattrnames.append( 'group' )
    pointattrs = numpy.empty( (numpts, len(pointattrnames)) )

    pointattrs[:,0] = mwsdb['radius'][wanted]
    pointattrs[:,1] = mwsdb['opacity'][wanted]

    rgb565 = mwsdb['colorindex'][wanted]

    pointattrs[:,2] = ((rgb565&0xF800)>>11) / 31.99  # Cr
    pointattrs[:,3] = ((rgb565&0x07E0)>>6)  / 63.99  # Cg
    pointattrs[:,4] = ((rgb565&0x001F)>>0)  / 31.99  # Cb
    if withgroup:
        pointattrs[:,5] = mwsdb['group'][wanted]

    points = mwsdb['xyz'][wanted]
    cgpoints = numpy.empty( (numpts, 3) )
    Tcg_rot = Tcgal[:3, :3]
    Tcg_trans = Tcgal[3, 0:3]
    for i in range(numpts):
        cgpoints[i] = numpy.dot(points[i], Tcg_rot) + Tcg_trans

    doit, msg = maybefile(outfname)
    print("#", msg, outfname, "(%d particles from groups %s)" % (numpts, groups), flush=True)
    if doit:
        _ = bgeo.BGeoPolyWriter( outfname, polyattrnames=[], polyattrs=None, polyverts=None, points=cgpoints, pointattrnames=pointattrnames, pointattrs=pointattrs )

########

mwsdb = sdb.SdbReader( mwaysdbfile ).read()

groupsused = ignoregroups.copy()

for line in clumpdefs.split('\n'):
    m = clumppat.match(line)
    if m is None:
        continue

    clumpname, gcodes = m.groups()

    if '-' in gcodes:
        gfrom, gto = [int(s) for s in gcodes.split('-')]
        unwanted = set( range(gfrom,gto+1) ).intersection( ignoregroups )
        if unwanted != set():
            print("## Ignoring clump %s groups %d..%d since it includes ignored group(s) %s" % (clumpname, gfrom,gto, list(unwanted)), flush=True)
            continue
        groups = slice(gfrom, gto+1)
        wanted = (mwsdb['group'] >= gfrom) & (mwsdb['group'] <= gto)
        groupsused.update( range( gfrom, gto+1 ) )
    else:
        groupno = int(gcodes)
        if groupno in ignoregroups:
            print("## Ignoring clump %s since it contains ignored group %d" % (clumpname, groupno), flush=True)
            continue
        groups = groupno
        wanted = (mwsdb['group'] == groupno)
        groupsused.add(groupno)

    process_clump( mwsdb, wanted, outstem+clumpname, groups )


## Now process anything else listed in "allparts.pop" which we haven't gotten above.

fileswithgroup = {}

def bestnameforgroup(groupno):
    global fileswithgroup

    if groupno not in fileswithgroup:
        return "group%d" % groupno

    fdir = "/fe4/ghez2/data/mway"
    def fqual(f):
        ff = os.path.join( fdir, f )
        if not os.path.exists(ff):
            return -1
        if os.path.islink(ff):
            return 1
        return os.path.getmtime(ff)

    fileswithgroup[groupno].sort(key=fqual, reverse=True)

    return fileswithgroup[groupno][0].replace('.sdb','')


with open(mwaysdbpopfile) as mwpopf:
    for line in mwpopf.readlines():
        ss = line.split()
        if len(ss) == 0 or ss[0][0] == '#':
            continue

        groupno = int(ss[0])
        popsdb = ss[-1]

        # ignore group if already handled above
        if groupno in groupsused:
            continue

        if groupno in fileswithgroup:
            fileswithgroup[groupno].append( popsdb )
        else:
            fileswithgroup[groupno] = [ popsdb ]
        
groupsbyfile = {}
for groupno in fileswithgroup.keys():
    f = bestnameforgroup(groupno)
    if f in groupsbyfile:
        groupsbyfile[f].append(groupno)
    else:
        groupsbyfile[f] = [groupno]

for f in sorted(groupsbyfile.keys()):
    groups = groupsbyfile[f]
    wanted = (mwsdb['group'] == groups[0])
    for groupno in groups[1:]:
        wanted |= (mwsdb['group'] == groupno)
    process_clump( mwsdb, wanted, outstem + f, groups )
