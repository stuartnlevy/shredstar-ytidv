#! /usr/bin/env python

import numpy

class SdbReader:

    # particle types (ptype field below):
    POINT = 0        # gaussian point with glow given by brightness (magnitude) and color
    BRIGHT_CLOUD = 1 # with given brightness (magnitude), color and radius
    DARK_CLOUD = 2   # with given opacity and radius, no glow
    BOTH_CLOUD = 3   # both bright and dark together
    SPIKE = 4        # sharp point with glow given by brightness & color

    typecode = { POINT:'P', BRIGHT_CLOUD:'C', DARK_CLOUD:'D', BOTH_CLOUD:'B', SPIKE:'S' }

    # Fields, from a C structure
    sdbrec = numpy.dtype( 
                [
                  ('xyz', ('>f4', 3)),
                  ('dxyz', ('>f4', 3)),
                  ('magnitude', '>f4'),
                  ('radius', '>f4'),
                  ('opacity', '>f4'),
                  ('id', '>i4'),
                  ('colorindex', '>i2'),
                  ('group', 'i1'),
                  ('ptype', 'i1')
                ])

    def __init__(self, infile):
        if hasattr(infile, 'read'):
            self.infile = infile
            self.infname = None
        elif infile == '-':
            self.infile = open('/dev/stdin', 'rb')
            self.infname = '-'
        else:
            self.infile = open(infile, 'rb')
            self.infname = infile

    @staticmethod
    def mag2lum(mag):
        return numpy.exp(-0.921034*(17+mag))
    # 5 magnitudes = factor of 100 brightness.  0.921... = log(100 ** (1/5))

    def __del__(self):
        if self.infname is not None:
            self.infile.close()

    def read(self, max=-1):
        if self.infile.seekable():
            return numpy.fromfile( self.infile, dtype=self.sdbrec, count=max )
        else:  # fromfile fails if reading from a pipe
            bytes = self.infile.read( -1 if max<0 else max*self.sdbrec.itemsize )
            return numpy.frombuffer( bytes, dtype=self.sdbrec )


    # From this C structure, from .../stardef.h :
    # typedef enum {ST_POINT, ST_BRIGHT_CLOUD ,ST_DARK_CLOUD, ST_BOTH_CLOUD, ST_SPIKE, ST_OFF} stype;
    #
    # typedef struct {
    #         float  x, y, z;
    #         float  dx, dy, dz;
    #         float  magnitude, radius;
    #         float  opacity;
    #         int  num;
    #         unsigned short  color;
    #         unsigned char   group;
    #         unsigned char   type;
    # }  db_star;

### Example code:

if __name__ == "__main__":
    sdb = SdbReader( '/ts/brobertson/virdir/sdb67b/1000.sdb' )
    pcles = sdb.read()

    for i in range( min( 10, len(pcles) ) ):
        p = pcles[i]
        print("pos:", p['xyz'], "vel:", p['dxyz'], "mag:", p['magnitude'], "lum:", sdb.mag2lum(p['magnitude']),"r:", p['radius'], "cindex:", p['colorindex'], 'type:', sdb.typecode[p['ptype']])

    # or you could access a whole slice of particles at once, e.g.
    # pcles = sdb.read()
    # positions = pcles['xyz']  # returns N x 3 array of all their positions
    # luminosities = sdb.mag2lum( pcles['magnitude'] )  # all their luminosities, etc.
