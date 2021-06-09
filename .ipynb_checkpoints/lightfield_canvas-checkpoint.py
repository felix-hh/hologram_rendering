from ipywidgets import IntSlider, FloatSlider, HBox, VBox, Output
from ipywidgets import Image as IpyImage
from ipycanvas import Canvas
from PIL import Image
import io
import numpy as np
import cv2

#this has to be defined before DisplayLF
# out = Output(layout={'border': '1px solid black', 'width': '200px', 'height': '140px'}) 
out = Output(layout={'border': '1px solid black'})

class DisplayLF(Canvas):
    def __init__(self, lf, width= None, height= None, sensitivity=2):
        """
        Initialize with a lightfield in np.uint8 format.
        Optionally pass width and height of the canvas and sensitivity to the mouse.
        """
        super().__init__(size=(width, height))
        
        # Adjustable dimensions of the canvas and mouse sensitivity
        self.width = width or lf.shape[-2]
        self.height = height or lf.shape[-3]
        self.dim = lf.ndim
        self.sensitivity = sensitivity
        self.display_image = IpyImage()
        out_width = width//2
        out_height = 3*height//(4)
        slider_height = height - out_height
        self.out = Output(layout={'border': '1px solid black', 'width': f'{out_width}px', 'height': f'{out_height}px'})
        
        # Prepare the lightfields to be rendered with the self.draw function()
        self.lf = self.convert_lf(lf)
        self.downsampled_lf = self.downsample_lf(self.lf)
        
        if self.dim == 6:
            self.u_min, self.u_max = 0, self.lf.shape[1]-1
            self.v_min, self.v_max = 0, self.lf.shape[2]-1
            self.z_min, self.z_max = 0, self.lf.shape[0]-1
        elif self.dim == 5:
            self.u_min, self.u_max = 0, self.lf.shape[0]-1
            self.v_min, self.v_max = 0, self.lf.shape[1]-1
            self.z_min, self.z_max = 0, 0
        elif self.dim == 4:
            self.u_min, self.u_max = 0, 0
            self.v_min, self.v_max = 0, 0
            self.z_min, self.z_max = 0, self.lf.shape[0]-1
        else:
            raise Exception("unexpected number of dimensions for lightfield")
        
        self.slider = IntSlider(min=0, max=self.z_max, value=0, description= 'Z',
                               layout={'width': '', 'height': f'{slider_height}px',
                                       'flex-flow': 'column', 'align-self': 'left'})
        self.slider.observe(self.handle_zoom_change, names='value')

        self.dragging = False
        self.x_mouse = None
        self.y_mouse = None
        
        self.u_idx = self.u_max//2
        self.v_idx = self.v_max//2
        self.z_idx = 0
        
        #Initialize state
        self.draw()
            
        self.on_mouse_down(self.mouse_down_handler)
        self.on_mouse_move(self.mouse_move_handler)
        self.on_mouse_up(self.mouse_up_handler)
        self.on_mouse_out(self.mouse_out_handler)
        
    @property
    def caption_uv_position(self):
        return f"u_idx = {int(self.u_idx)}\tv_idx = {int(self.v_idx)}"
        
    @property
    def caption_zoom(self):
        return f"z_idx = {self.z_idx}"
    
    @property
    def position(self):
        if self.dim >= 6:
            return tuple((int(self.z_idx), int(self.u_idx), int(self.v_idx)))
        elif self.dim == 5:
            return tuple((int(self.u_idx), int(self.v_idx)))
        elif self.dim == 4:
            return int(self.z_idx)
        else:
            raise Exception("unexpected dimensionality while obtaining display position")
    def convert_lf(self, lf):
        """
        Converts lightfield to np.uint8 with a white background and no alpha channel.
        """
        lf = np.float32(lf)
        if np.max(lf) > 1:
            lf = lf/256
        lf = np.uint8(lf*256)
        
        if lf.shape[-1] > 3:
            lf[lf[:,:,:,:,3] == 0] = (255,255,255,0) #convert alpha to white. 
            lf = lf[:,:,:,:,:3]
        # while lf.ndim < 6:
            # lf = np.expand_dims(lf, 0)
            
        lf = resize_lightfield(lf, (self.width, self.height))    
        return lf
    
    def downsample_lf(self, converted_lf):
        """
        Creates a downsampled version of the lightfield for fast rendering when dragging
        """
        #For simplicity, we use every xth and yth pixel, instead of more complicated downsampling methods.
        DOWNSAMPLE_DIMS = (150, 150) # make output shape of each image less than these dimensions. 
        step_size_x = int(converted_lf.shape[2]/DOWNSAMPLE_DIMS[0])+1
        step_size_y = int(converted_lf.shape[2]/DOWNSAMPLE_DIMS[0])+1
        
        return converted_lf[...,::step_size_x, ::step_size_y, :]
    
    def show(self):
        return HBox((self, VBox((self.out, self.slider))))

    def draw(self):
        # Log u,v indices in canvas
        with self.out:
            self.out.clear_output()
            self._print_state()
        
        # Select image data from lightfields
        if self.dragging:
            img_data = self.downsampled_lf[self.position]
        else:
            img_data = self.lf[self.position]
        
        # Write to canvas
        self.display_image.value = numpy_to_image_widget_value(img_data)
        self.draw_image(self.display_image, 0, 0, self.width, self.height)

    def mouse_down_handler(self, pixel_x, pixel_y):
        self.dragging = True
        self.x_mouse = pixel_x
        self.y_mouse = pixel_y

    def mouse_move_handler(self, pixel_x, pixel_y):
        if self.dragging:
            delta_x = pixel_x-self.x_mouse
            delta_y = pixel_y-self.y_mouse
            
            self.x_mouse = pixel_x
            self.y_mouse = pixel_y
            delta_u = (delta_y*self.u_max/self.height)*self.sensitivity
            delta_v = (delta_x*self.v_max/self.width)*self.sensitivity
            
            self.u_idx = np.clip(delta_u + self.u_idx, self.u_min, self.u_max)
            self.v_idx = np.clip(-delta_v + self.v_idx, self.v_min, self.v_max)
            
            self.draw()
    
    def mouse_up_handler(self, pixel_x, pixel_y):
        if self.dragging:
            self.dragging = False
            self.draw()
    
    def mouse_out_handler(self, pixel_x, pixel_y):
        if self.dragging:
            self.dragging = False
            self.draw()
        self.mouse_move_handler(pixel_x, pixel_y)
        
    def handle_zoom_change(self, change):
        self.z_idx = change.new
        with self.out:
            self.out.clear_output()
            self._print_state()
        self.draw()
            
    def _print_state(self):
        with self.out:
            print(self.caption_uv_position)
            print(self.caption_zoom)
        
def numpy_to_image_widget_value(data, format="png", quality=75):
    buffer = io.BytesIO()
    image = Image.fromarray(data)
    image.save(buffer, format= format, quality= quality)
    return buffer.getvalue()

def resize_lightfield(raw_lightfield, desired_size): # -> lightfield
    """
    Applies cv2.resize to each image of the lightfield (6-dim)
    input:
    raw_lightfield to be resized
    desired_size of images in the output in width X height tuple format
    output:
    lightfield with appropiate image size
    """
    if raw_lightfield.ndim == 6:
        n_depth, n_cols, n_rows, *_ = raw_lightfield.shape
        output_lightfield = np.empty([n_depth, n_cols, n_rows] + list(desired_size)[::-1] + [3], dtype=np.uint8)


        for k, depth in enumerate(raw_lightfield):
            for i, row in enumerate(depth):
                for j, image_data in enumerate(row):
                    resized_image_data = cv2.resize(image_data, desired_size)
                    output_lightfield[k, i, j] = resized_image_data
    elif raw_lightfield.ndim == 5:
        n_cols, n_rows, *_ = raw_lightfield.shape
        output_lightfield = np.empty([n_cols, n_rows] + list(desired_size)[::-1] + [3], dtype=np.uint8)


        for i, row in enumerate(raw_lightfield):
            for j, image_data in enumerate(row):
                resized_image_data = cv2.resize(image_data, desired_size)
                output_lightfield[i, j] = resized_image_data
        
    elif raw_lightfield.ndim == 4:
        n_rows, *_ = raw_lightfield.shape
        output_lightfield = np.empty([n_rows] + list(desired_size)[::-1] + [3], dtype=np.uint8)

        for j, image_data in enumerate(raw_lightfield):
            resized_image_data = cv2.resize(image_data, desired_size)
            output_lightfield[j] = resized_image_data
                
    return output_lightfield