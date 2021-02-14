import matplotlib.pyplot as plt
import os
import imageio
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

def spline_interp(rnge,n_interp=50,n_points=4):
    np.random.seed(seed=42)
    y = np.random.uniform(rnge[0],rnge[1],size=(n_points,))
    # y = np.sort(y)
    x = np.linspace(0,100,n_points,endpoint=True)
    x_new = np.linspace(0,100,n_interp,endpoint=True)
    y_new = np.interp(x_new,x,y)
    return y_new
    
    
def save_lf(lf,frame_num,DATA_DIR,LF_name):
    if not os.path.exists(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num)):
        os.mkdir(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num))
    n_lf = lf.shape[0]
    for k in range(n_lf):
        img = lf[k,...,:3]*255.
        img = img.astype('uint8')
        imageio.imwrite(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num,'view_%.6d.png'%k),img)

def save_lf(lf,frame_num,DATA_DIR,LF_name):
    if not os.path.exists(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num)):
        os.mkdir(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num))
    n_lf = lf.shape[0]
    angular = int(np.sqrt(n_lf))
    spatial = lf.shape[-2]
    lf = lf.reshape([angular,angular,spatial,spatial,4])
    for k in range(0,angular):
        for l in range(0,angular):
            img = lf[k,l,...,:3]*255.
            img = resize(img,[100,100,3],order=1,preserve_range=True,anti_aliasing=True)
            img = img.astype('uint8')
            # img = zoom(img,[0.2,0.2,1.0],prefilter=False)
            imageio.imwrite(os.path.join(DATA_DIR,LF_name,'frame_%.3d'%frame_num,'view_%.6d_%.6d.png'%(k,l)),img)

def save_frame(lf,frame_num,DATA_DIR,LF_name):
    if not os.path.exists(os.path.join(DATA_DIR,LF_name)):
        os.mkdir(os.path.join(DATA_DIR,LF_name))
    n_lf = lf.shape[0]
    for k in range(n_lf):
        img = lf[k,...,:3]*255.
        img = img.astype('uint8')
        imageio.imwrite(os.path.join(DATA_DIR,LF_name,'view_%.6d.png'%frame_num),img)

def euler2RotMat(z,y,x):
    # takes as input 3 rotation angles in degrees
    # in the order z rot, y rot and x rot
    # return a 3x3 rotation matrix
    r = Rotation.from_euler('zyx', [z,y,x], degrees=True)
    return Rotation.as_matrix(r)


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9)
    )
    bleed = 0
    fig.subplots_adjust(
        left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
    )

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
