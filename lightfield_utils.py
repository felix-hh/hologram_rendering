import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import tqdm



def load_lightfield_from_folder(folder_path, images_per_row, file_suffix= ".tif"):
    """
    Generates a lightfield from a folder of images
    The names of the images must be such that everything is in order of capture, as if the images had been
    taken on a zig zag path with the camera. May need to play around with move axis and reversing indices.
    
    input: 
    folder_path to folder containing only images that can be sorted without key
    images_per_row number of images per row of the lightfield.
    
    output: 
    lightfield of shape n_columns X n_rows X image.shape
    """
    image_paths = sorted(list(Path(folder_path).iterdir()))
    image_paths = [path for path in image_paths if str(path).endswith(file_suffix)]
    result = load_lightfield_from_paths(image_paths, images_per_row)
    
    return result     

def load_lightfield_from_paths(image_paths, images_per_row, resize= None, transform= None):
    """
    Generates a lightfield from a sorted list of filepaths
    The order of paths must be such that everything is in order of capture, as if the images had been
    taken on a zig zag path with the camera. May need to play around with move axis and reversing indices.
    
    input: 
    image_paths: sorted path to each of the images to be loaded
    images_per_row number of images per row of the lightfield.
    resize: target size of the images to be loaded, if any (width, height)
    transform: a transformation to be applied before the image is resized (i.e. crop before resize to preserve
    resolution)
    
    output: 
    lightfield of shape n_columns X n_rows X image.shape
    """
    n_images = len(image_paths)
    images_per_column = int(n_images / images_per_row)
    
    image_dimensions = list(np.array(Image.open(image_paths[0])).shape)
    if resize is not None:
        image_dimensions = list(resize)[::-1] + [3]
        
    lightfield = np.empty([images_per_column, images_per_row] + image_dimensions, dtype=np.uint8)
    
    for image_idx, image_path in enumerate(tqdm.tqdm(image_paths)):
        v_idx, u_idx = image_idx%images_per_row, image_idx//images_per_row
        
        if u_idx %2 == 0: # maintain u_idx in different zig-zag directions left-right and right-left.
            v_idx = images_per_row - v_idx -1
            
        image_data = np.array(Image.open(image_path))
        if transform is not None:
            image_data = transform(image_data)
        if resize is not None:
            image_data = cv2.resize(image_data, resize)
        lightfield[u_idx, v_idx] = image_data if image_data.dtype==np.uint8 else (image_data*256).astype(np.uint8)
    
    return lightfield  

def load_lightfield(filepath):
    """
    Loads lightfield from an npy file
    input
    filepath: location of the .npy file
    output
    lf: np array of the lightfield. 
    """
    
    lf = np.load(filepath)
    lf = np.nan_to_num(lf)
    
    #convert float to np.uint8
    lf = lf*256 if lf.dtype == np.float32 else lf
    lf = lf.astype(np.uint8)
    
    print("Loaded lightfield with dtype:", lf.dtype, "and shape:", lf.shape)
    
    return lf

def get_average_lightfield(lf):
    """
    Returns an image of the "average" of all frames of the lightfield. Useful to detect aberrations or 
    problems during lightfield capture
    input:
    ligthfield: numpy array of the lightfield
    output:
    avg_lf: a np.uint8 image of the average of all frames
    """
    u_num, v_num, *_ = lf.shape
    
    lf = lf*256 if lf.dtype == np.float32 else lf
    lf = lf.astype(np.uint8)
    
    sum_lf = np.sum(lf, axis=0)
    sum_lf = np.sum(sum_lf, axis=0)
    
    avg_lf = sum_lf/(u_num*v_num)
    avg_lf = avg_lf.astype(np.uint8)
    # print("Average lightfield shape:", avg_lf.shape)
    
    return np.uint8(avg_lf)

def crop_to_shape(lightfield, new_shape):
    """
    Crops every image evenly from both sides of the lightfield to the desired shape
    """
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



def resize_lightfield(raw_lightfield, desired_size): # -> lightfield
    """
    Applies cv2.resize to each image of the lightfield
    input:
    raw_lightfield to be resized
    desired_size of images in the output in width X height tuple format
    output:
    lightfield with appropiate image size
    """
    n_cols, n_rows, *_ = raw_lightfield.shape
    output_lightfield = np.empty([n_cols, n_rows] + list(desired_size)[::-1] + [3])
    
    
    for i, row in enumerate(raw_lightfield):
        for j, image_data in enumerate(row):
            resized_image_data = cv2.resize(image_data, desired_size)
            output_lightfield[i, j] = resized_image_data
            
    return output_lightfield

def save_rendered_views(image_folder, rendered_lightfield):
    """
    Saves all the images of a lightfield in a folder
    """
    
    image_folder.mkdir(parents= True, exist_ok=True)

    for row_idx, row in enumerate(tqdm.tqdm(rendered_lightfield)):
        for column_idx, image_data in enumerate(row):
            if image_data.dtype != np.uint8:
                image_data = np.uint8(image_data*256)
                print("Converting np.float32 np.uint8. Use np.uint8 dtype to save rendered views")
            image = Image.fromarray(np.uint8(image_data))
            image.save(image_folder / f"{row_idx:04d}_{column_idx:04d}.png", format= "png", opt= "yes", quality= 75)

    print("Saved at", str(image_folder.absolute()))
