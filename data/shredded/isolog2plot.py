#! /usr/bin/env python3

import sys, os
import re

quantpat = re.compile('vmax (\S+), vthresh (q\S+) => (\S+), for .* (\d\d\d\d)\s*$')
# vmax 0.441112, vthresh q1 => 1e-11, for /sd3/ghez2/data/shredded/isobhb5_300/b5_300 0610


infs = []
outstem = 'isoplot'
quants = None

ii = 1
while ii<len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-o':
        outstem = sys.argv[ii]; ii += 1
    elif opt == '-i':
        infs.append( sys.argv[ii] ); ii += 1
    elif opt == '-q':
        ss = sys.argv[ii].replace(',',' ').split()
        quants = {}
        for s in ss:
            if s.startswith('q'):
                quants[s] = float('0.' + s[1:])
            else:
                qstr = ("q" + s).replace('q0.','q')   # 0.0025 => q0025
                quants[qstr] = s


qvals = { 'q0':{} }

sval = None
for infname in infs + sys.argv[ii:]:
    with open('/dev/stdin' if infname=='-' else infname) as inf:
        for line in inf.readlines():
            m = quantpat.match(line)
            if m:
                (svmax, qstr, sval, sframeno) = m.groups()
                qvals['q0'][sframeno] = svmax
                if qstr not in qvals:
                    qvals[qstr] = {}
                qvals[qstr][sframeno] = sval

qplottags = []

if sval is None:
    print("No data -- not writing to %s.dat + %s.gnuplot" % (outstem,outstem))
    sys.exit(1)

print("Writing to %s.gnuplot + %s.dat" % (outstem,outstem))
with open(outstem+'.dat', 'w') as datf, open(outstem+'.gnuplot', 'w') as pltf:
    for qstr in sorted(qvals.keys()):
        if qstr=='q0' or quants is None or qstr in quants:
            qplottags.append( qstr )

            print(" =%s=" % qstr, file=datf)
            for frameno in sorted(qvals[qstr].keys()):
                print("%s %s" % (frameno, qvals[qstr][frameno]), file=datf)
            print("", file=datf)

            title = 'max' if (qstr=='q0') else qstr
            print("plot '%s.dat' index '=%s=' title '%s'" % (outstem, qstr, title), file=pltf)
                       
