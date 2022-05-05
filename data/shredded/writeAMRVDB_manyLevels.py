#! /usr/bin/env python3
import sys, os
import pyopenvdb as vdb
import yt
import numpy as np
import re
import time

#########################################################################
### Modify these values to point to your own data on your own machine ###
#########################################################################

dataFilePath = None # "/fe0/deslsst/renaissance/normal/RD0074/RedshiftOutput0074"
outFileDir   = None # "/fe2/amr/data/"
variable = "density"
isFlash = False
takeLog = False
scale = 1.0 # 2**5
resume = False

# More advanced parameters here
minLevel = 0
maxLevel = None # ds.index.max_level if not overridden by -maxlevel opt
frameno = None

framepat = re.compile('_(\d\d\d\d)$')

ii = 1
while ii < len(sys.argv) and sys.argv[ii][0] == '-':
    opt = sys.argv[ii]; ii += 1
    if opt == '-i':
        dataFilePath = sys.argv[ii]; ii += 1
    elif opt == '-o':
        outFileDir = sys.argv[ii]; ii += 1
    elif opt == '-log':
        takeLog = True
    elif opt == '-v':
        variable = sys.argv[ii]; ii += 1
    elif opt == '-l' or opt == '-maxlevel':
        maxLevel = int(sys.argv[ii]); ii += 1
    elif opt == '-r':
        resume = True
    

if ii != len(sys.argv) or dataFilePath is None:
    print("""Usage: %s [-i AMR_data_file] [-o outdir] [-log] [-v varname] [-l maxlevel] [-r]
Read a yt AMR data file, write a series of datafiles of form <outdir>/<varname>_ren_level<level>.<NNNN>.vdb
With -r, only writes output files if they don't already exist.""" % sys.argv[0])
    sys.exit(1)

#########################################################################
#########################################################################

if dataFilePath is None:
    print("Must specify -i yt_input_dataset", file=sys.stderr)
    sys.exit(1)
# Load the dataset
ds = yt.load(dataFilePath)

if maxLevel is None:
    maxLevel = ds.index.max_level

m = framepat.search( dataFilePath )
if m is None:
    framestr = ''
else:
    framestr = '.' + m.group(1)   # .NNNN

# Keep track of level 0 voxel size
largestVSize = None
smallestVSize = scale*1. / (ds.domain_dimensions*ds.refine_by**maxLevel)

# Error checking: is this variable in the data?
if not [item for item in ds.field_list if item[1] == variable]:
    print("ERROR: Unknown field-variable name: " + variable, file=sys.stderr)
    print("")
    print("Available variables (for -v option): ", " ".join([item[1] for item in ds.field_list if not item[1].startswith('particle_')]))
    sys.exit(1)

# This is required to be able to write out ghost zones for FLASH data
#if isFlash:
#    ds.periodicity = (True, True, True)
if hasattr(ds, 'force_periodicity'):
    ds.force_periodicity()  # yt >=4.1
else:
    ds.periodicity = (True, True, True)

try:
    if not os.path.isdir(outFileDir):
        os.makedirs(outFileDir)
except:
    pass

if not os.path.isdir(outFileDir):
    print("Couldn't create output directory %s" % outFileDir)
    sys.exit(1)

# Iterate through all levels
for level in range(minLevel, maxLevel+1):

    # Calculate a reasonable voxel size
    resolution = ds.domain_dimensions*ds.refine_by**level
    vSize = scale*1/float(resolution[0])

    # Keep track of level 0 voxel size
    if level==minLevel:
        largestVSize = vSize


    outFilePath = "%s/%s_ren_level%d%s.vdb" % (outFileDir, variable.strip(), level, framestr)  # remove any blanks from variable name: 'c12 ' => 'c12'

    if resume and os.path.exists(outFilePath):
        print("Skipping level-%d file %s" % (level, outFilePath), flush=True)
        continue

    t0 = time.time()

    # Select specific level of grids set from dataset
    gs = ds.index.select_grids(level)

    # Initiate OpenVDB FloatGrid
    maskCube = vdb.FloatGrid()
    dataCube = vdb.FloatGrid()

    # Go over all grids in current level
    for index in range(len(gs)):

        subGrid = gs[index]

        # Extract grid (without ghost zone) with specific varible
        subGridVar = subGrid[variable]

        # Extract grid (with ghost zone) with specific varible
        subGridVarWithGZ = subGrid.retrieve_ghost_zones(n_zones=1, fields=variable)[variable]

        # Take the log (base 10), if needed
        if takeLog:
            subGridVarWithGZ = np.log10(subGridVarWithGZ)

        # Extract mask grid (eg. {[1 0 0 1],[0 1 0 1]...})
        mask = subGrid.child_mask

        # ijkout is the global x,y,z index in OpenVDB FloatGrid
        # provided as int64's, but pyopenvdb requires int32 indexes.
        ijkout = subGrid.get_global_startindex().astype(np.int32)

        # Copy data from grid to OpenVDB FloatGrid starting from global x,y,z index in OpenVDB FloatGrid
        maskCube.copyFromArray(mask, ijk=(ijkout[0],ijkout[1],ijkout[2]))
        dataCube.copyFromArray(subGridVarWithGZ, ijk=(ijkout[0],ijkout[1],ijkout[2]))    
    
    # Scale and translate
    dataMatrix = [[vSize, 0, 0, 0], [0, vSize, 0, 0], [0, 0, vSize, 0], [-vSize/2, -vSize/2, -vSize/2, 1]]
    maskMatrix = [[vSize, 0, 0, 0], [0, vSize, 0, 0], [0, 0, vSize, 0], [ vSize/2,  vSize/2,  vSize/2, 1]]
    dataCube.transform = vdb.createLinearTransform(dataMatrix) 
    maskCube.transform = vdb.createLinearTransform(maskMatrix)

    # Write out the generated VDB
    output = []
    dataCube.name = "density"
    maskCube.name = "mask"
    output.append(maskCube)
    output.append(dataCube)

    
    vdb.write(outFilePath+'+', output, {'translated':'[0, 0, 0]'})
    os.rename(outFilePath+'+', outFilePath)

    print("Wrote level-%d file %s in %d ms" % (level, outFilePath, (time.time() - t0)*1000), flush=True)
