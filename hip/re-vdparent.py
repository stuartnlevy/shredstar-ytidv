#! /usr/bin/env python

import sys, os

import virdirp

camhook = "/obj/CAM_OUT"

temppath = "re-vdparent.tmp.wf"

center = "0,0,0"
order = "xyz"  # application order for rotations in target nodes.  Houdini default is xyz.  Virdir default is zxy.

vdparented = f"vdparented -order {order} -z0  -p {center}"

"""
vdparented [-p centerX,Y,Z] [-uprxyz Rx,Ry,Rz] [-orient Rx,Ry,Rz] [-z0] [-y0] [-x0] [-o outstem] [-order zxy] [-v]  [infile.vd-or-wf]
Transforms a virdir path, either in .vd or .wf form,
into "parented" form: as the composition of two paths, in the sense (if infile => Tc2w):
    Tc2w = Tc2p * Tp2cen * Tcen2w
where:
    Tcen2w is a pure translation to the given centerX,Y,Z point
    Tp2cen is a rotation which moves the camera-to-center to the -Z axis,
	plus a translation along Z to the camera's distance from the center,
    Tc2p is a pure rotation of the camera relative to the above
Writes out 3 new paths,
    <stem>.c2p.<suf> and <stem>.p2cen.<suf> and <stem>.cen2w.<suf>
"""

def reparent_bottom_two(camnode, netpathstem=None):

    midnode = camnode.input(0)

    lastkeyframe = max( virdirp._lastkeyframe(camnode.path()), virdirp._lastkeyframe(midnode.path()) )

    if netpathstem is not None:
        netwf = netpathstem.replace('.wf','') + ".before.wf"
        virdirp.exportCamera( camnode.path(), netwf, endFrame=lastkeyframe )
        print('"before" path written to', netwf)

    midparent = midnode.input(0)
    midnode.setInput(0, None)  # Temporarily detach cam stack from its parent blend node

    virdirp.exportCamera(camnode.path(), temppath, endFrame=lastkeyframe)

    stem = temppath.replace('.wf', '')

    os.system( vdparented + f" -o {stem}  {temppath}" )

    # now resulting files are {stem}_{order}.<tag>.wf
    # We want to put keys in the same time points as the originals.

    for node, tag in (camnode, 'c2p'), (midnode, 'p2cen'):
        fname = f"{stem}_{order}.{tag}.wf"
        virdirp.updatePath( fname, node.path() )


    # reattach
    midnode.setInput(0, midparent)

    # write out overall path again after the update

    if netpathstem is not None:
        netwf = netpathstem.replace('.wf','') + ".after.wf"
        virdirp.exportCamera( camnode.path(), netwf, endFrame=lastkeyframe )
        print('"after" path written to', netwf)



if __name__ == "__main__":
    scenefile = sys.argv[1]
    testpathstem = sys.argv[2] if len(sys.argv) > 2 else None
    hou.hipFile.load( scenefile, ignore_load_warnings=True )
    camhooknode = hou.node(camhook)
    if camhooknode is None:
        print("%s: Can't find node %s in scene file %s" % (camhook, scenefile))

    reparent_bottom_two( camhooknode.input(0), testpathstem )

    outscenefile = scenefile.replace('.hipnc','') + '.revdparent.hipnc'
    hou.hipFile.save( outscenefile )
