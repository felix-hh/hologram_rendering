import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from lightfield_canvas import DisplayLF

# The dataset contains images captured with a camera following a zig-zag path, where the camera moves on
# horizontal lines, and when complete, it shifts up one level and repeats in the opposite direction.
# The name of each image starts with 4 numbers, that represent the index of the image when collected.
# The index of the image on the horizontal line is the u coordinate. The index in the vertical line is the v coordinate
data_folder = os.path.join(os.getenv("FELIX_DATA"), "LensCalibrated_6s","LensCalibrated_6s","Registered-ForFelix")

# Each folder has N_IMAGES
N_IMAGES = len(os.listdir(data_folder))-1

# Each v line has IMAGES_PER_LINE images in the u axis, and there are IMAGES_PER_VERTICAL in v.
IMAGES_PER_LINE = 13
IMAGES_PER_VERTICAL = N_IMAGES // IMAGES_PER_LINE
IMAGE_DIMENSIONS = (1770, 1476)
DTYPE = np.uint8
DESIRED_DIMENSIONS = (572, 440)
if not DESIRED_DIMENSIONS:
    DESIRED_DIMENSIONS = IMAGE_DIMENSIONS

# We load a lightfield with the images in the folder.
# Initialize lightfield
lightfield = np.ones([IMAGES_PER_LINE, IMAGES_PER_VERTICAL, DESIRED_DIMENSIONS[1], DESIRED_DIMENSIONS[0], 3], dtype= DTYPE)
# Iterate through each image name in the folder
completed_count = 0
for image_name in os.listdir(data_folder):
    # If the image name is valid
    if image_name[:4].isnumeric():
        # Put image in the lightfield array, resize image if necessary
        image_idx = int(image_name[:4])
        u_idx, v_idx = image_idx%IMAGES_PER_LINE, image_idx//IMAGES_PER_LINE
        if v_idx %2 == 0: # maintain u_idx in different zig-zag directions left-right and right-left.
            u_idx = IMAGES_PER_LINE - u_idx -1
        image_data = np.array(Image.open(os.path.join(data_folder, image_name)))
        image_data = cv2.resize(image_data, DESIRED_DIMENSIONS)
        lightfield[u_idx, v_idx] = image_data

        #log for debugging
        completed_count += 1
        if completed_count % IMAGES_PER_LINE == 0:
            print(f"Loaded image {completed_count} out of {N_IMAGES}")

    #If the index is not a number, continue
    else:
        print("Ignored file: " + image_name)
        continue

lightfield = np.moveaxis(lightfield, [0,1,2,3,4],[1,0,2,3,4]) # DisplayLF is [vertical, horizontal, vertical, horizontal, rgb]
print("Lightfield ready")

import torch
import tqdm
import pytorch3d

from pytorch3d.renderer.cameras import SfMPerspectiveCameras, OpenGLOrthographicCameras
from pytorch3d.renderer import look_at_view_transform
import LightfieldViewer as LV

# Set the cuda device
device = torch.device("cuda:2")
torch.cuda.set_device(device)
dtype = torch.uint8

### CONFIG ###
to_render_lightfield = np.array(lightfield, dtype= np.float32)/255 #divide by 255 since original format is uint8.
dtype = torch.float32 # must be float
zsep = -.1; # the separation between xy and uv planes
output_dimensions = (572, 440); # number of pixels in rendererd views
Np = max(output_dimensions)
Nv = 101; # the number of vertices across for the planar mesh

### ZOOM ###
cam_dist = 2

### ANGLE ###
angle_min = -20
angle_max = 20
angle_views = 384

### ELEVATION ###
elev_min = -20
elev_max = 20
elev_views = 96

### CAMERA ###
asp = 1;
focal_length = [1, 1/asp];
principal_point = [0, 0]

views = 20;
pp = torch.tensor(principal_point).expand(views,2)

#Future: Can  use different distances for zooming. would be 6-dim array :)
#zoom_min
#zoom_max
#zoom_views

def pad_to_shape(lightfield, new_shape):
    _, _, current_height, current_width, _ = lightfield.shape
    new_width, new_height = new_shape

    def get_pad(old_dim, new_dim):
        return (int(np.ceil((new_dim-old_dim)/2)), (new_dim-old_dim)//2)

    width_pad = get_pad(current_width, new_width)
    height_pad = get_pad(current_height, new_height)

    print (width_pad, height_pad)

    if not all([value >= 0 for pad_tuple in (width_pad, height_pad) for value in pad_tuple]):
        raise Exception("The requested padded shape is smaller than the current shape, but padding can only increase dimensions")

    lightfield = np.pad(lightfield, ((0,0), (0,0), height_pad, width_pad, (0,0)))

    return lightfield

to_render_lightfield = pad_to_shape(to_render_lightfield, (Np, Np))

def get_RTs(angles, elevs, cam_dist):
    """
    Returns the rotation and translation transformations for an array of cameras
    on a meshgrid of angles x elevs

    Input:
    angles: numpy array of angles to sample. Nx array
    elevs: numpy array of elevations to sample. Ny array
    cam_dist: the distance of the camera to the object

    Output:
    Rs: rotation transformations. (Nx, Ny, 3, 3)
    Ts: translation transformations. (Nx, Ny, 3)

    TODO: handle non-array values for angles, elevs
    """


    angle_views = len(angles)
    elev_views = len(elevs)

    dist = cam_dist*torch.ones(angle_views, dtype=dtype).view(angle_views) # distance from camera to the object
    angles = torch.tensor(angles, dtype=dtype).view(angle_views)  # angle of azimuth rotation in degrees
    elevs = torch.tensor(elevs, dtype=dtype).view(elev_views)

    elevs, angles = torch.meshgrid(elevs, angles)

    Rs = torch.empty((elev_views, angle_views, 3, 3))
    Ts = torch.empty((elev_views, angle_views, 3))

    for k in range(elev_views):
        Rs[k], Ts[k] = look_at_view_transform(dist, elevs[k], angles[k], device=device) # (views,3,3), (views,3)

    return Rs, Ts


def get_cameras(Rs, Ts, asp, focal_length):
    """
    Returns a list of PerspectiveCameras given Rs and Ts.
    """
    elev_views = Rs.shape[0]
    angle_views = Rs.shape[1]
    #generate focal length and principal point
    fl = cam_dist*torch.tensor(focal_length).expand(angle_views,2)
    pp = torch.tensor(principal_point).expand(angle_views,2)

    cameras = []

    for k in range(Rs.shape[0]):
        R, T = Rs[k], Ts[k]
        C = SfMPerspectiveCameras(focal_length= fl, principal_point= pp, R= R, T= T, device= device)
        cameras.append(C)

    return cameras

def get_rendered_views(lightfield, cameras, zsep, Np, Nv, show_progress= False):
    """
    Returns the rendered views for cameras at different angles and elevations.
    """
    results = []

    u_count, v_count, x_count, y_count, c = lightfield.shape

    for cams in tqdm.tqdm(cameras, disable= not show_progress):
        #create lightfieldViewer
        lighfieldViewer = LV.LightfieldViewerModel(device=device,dtype=dtype, init_cam=cams,
                                        zsep=zsep, Np=Np, Nv=Nv, scale=2)

        # the grid of lightfield coordinates
        u = np.linspace(-1,1,u_count) # Nu x 1 regular grid of values
        v = np.linspace(-1,1,v_count) # Nv x 1 regular grid of values
        x = np.linspace(-1,1,x_count) # Nx x 1 regular grid of values
        y = np.linspace(-1,1,y_count) # Ny x 1 regular grid of values

        renderedViews = lighfieldViewer(lightfield=lightfield, u=u, v=v, x=x, y=y)
        results.append(renderedViews)

    results = np.array(results[::-1])
    return results

angles = np.linspace(angle_min, angle_max, angle_views) # linspace of ashow_progress= degrees
elevs = np.linspace(elev_min, elev_max, elev_views) # linspace of elevs

Rs, Ts = get_RTs(angles, elevs, cam_dist)
cameras = get_cameras(Rs, Ts, asp, focal_length)

rendered_views = get_rendered_views(to_render_lightfield, cameras, zsep, Np, Nv, show_progress= True)
rendered_views = rendered_views[::-1]

def crop_to_shape(lightfield, new_shape):
    _, _, current_height, current_width, _ = lightfield.shape
    new_width, new_height = new_shape

    def get_slice(old_dim, new_dim):
        return ((old_dim - new_dim)//2, (old_dim - new_dim)//2 + new_dim)

    width_slice = get_slice(current_width, new_width)
    height_slice = get_slice(current_height, new_height)

    if not all([current_value >= new_value
                for current_value, new_value in zip((current_height, current_width), (new_height, new_width))]):
        raise Exception("The requested cropped shape is bigger than the current shape, but cropping can only decrease dimensions")

    (y0, y1), (x0, x1) = height_slice, width_slice

    lightfield = lightfield[:, :, y0 : y1, x0 : x1, :]
    return lightfield

cropped_rendered_views = crop_to_shape(rendered_views, output_dimensions)

filesize_original_lightfield = lightfield.itemsize*lightfield.size//1024**2 # MB size
filesize_rendered_lightfield = rendered_views.itemsize*rendered_views.size//1024**2 # MB size
print(f"The original lightfield occupies {filesize_original_lightfield} Mb")
print(f"The rendered lightfield occupies {filesize_rendered_lightfield} Mb")

output_filename = f"rendered_views_{elev_views}x{angle_views}x{output_dimensions[0]}x{output_dimensions[1]}.npy"

np.save(output_filename, cropped_rendered_views)

#Test output has been saved correctly
input_filename = f"rendered_views_{elev_views}x{angle_views}x{output_dimensions[0]}x{output_dimensions[1]}.npy"
rendered_lightfield = np.load(input_filename)
rendered_lightfield = np.nan_to_num(rendered_lightfield)

from pathlib import Path

def save_rendered_views():
    image_folder = Path(".") / "rendered_views" / input_filename[:-4]
    image_folder.mkdir(parents= True, exist_ok=True)

    for row_idx, row in enumerate(rendered_lightfield):
        for column_idx, image_data in enumerate(row):
            image = Image.fromarray(np.uint8(image_data*256))
            image.save(image_folder / f"{row_idx:04d}_{column_idx:04d}.png", format= "png", opt= "yes", quality= 75)

    print("Saved at", str(image_folder.absolute()))

save_rendered_views()
