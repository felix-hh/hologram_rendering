from ipywidgets import IntSlider, FloatSlider, HBox, VBox, Output
from ipywidgets import Image as IpyImage
from ipycanvas import Canvas
from PIL import Image
import io
import numpy as np
import cv2


#this has to be defined before DisplayLF
out = Output(layout={'border': '1px solid black', 'width': '200px'}) 

class DisplayLF(Canvas):
    def __init__(self, lf, width= 400, height= 400, sensitivity=2):
        """
        Initialize with a lightfield.
        Optionally pass width and height of the canvas and sensitivity to the mouse.
        """
        super().__init__(size=(width, height))
        
        # Adjustable dimensions of the canvas and mouse sensitivity
        self.width = width
        self.height = height
        self.sensitivity = sensitivity
        self.display_image = IpyImage()
        
        # Prepare the lightfields to be rendered with the self.draw function()
        self.lf = self.convert_lf(lf)
        self.downsampled_lf = self.downsample_lf(self.lf)
        
        self.u_min, self.u_max = 0, self.lf.shape[0]-1
        self.v_min, self.v_max = 0, self.lf.shape[1]-1

        self.dragging = False
        self.x_mouse = None
        self.y_mouse = None
        
        self.u_idx = self.lf.shape[0]//2
        self.v_idx = self.lf.shape[1]//2
        
        self.draw()
            
        self.on_mouse_down(self.mouse_down_handler)
        self.on_mouse_move(self.mouse_move_handler)
        self.on_mouse_up(self.mouse_up_handler)
        self.on_mouse_out(self.mouse_out_handler)
        

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
            
        return lf
    
    def downsample_lf(self, converted_lf):
        """
        Creates a downsampled version of the lightfield for fast rendering when dragging
        """
        #For simplicity, we use every xth and yth pixel, instead of more complicated downsampling methods.
        DOWNSAMPLE_DIMS = (150, 150) # make output shape of each image less than these dimensions. 
        step_size_x = int(converted_lf.shape[2]/DOWNSAMPLE_DIMS[0])+1
        step_size_y = int(converted_lf.shape[2]/DOWNSAMPLE_DIMS[0])+1
        
        return converted_lf[:,:,::step_size_x, ::step_size_y, :]
    
    def show(self):
        return HBox((self, out))

    @out.capture()
    def draw(self):
        # Log u,v indices in canvas
        with out:
            out.clear_output()
            print(f"u_idx = {int(self.u_idx)}\tv_idx = {int(self.v_idx)}")
        
        # Select image data from lightfields
        if self.dragging:
            img_data = self.downsampled_lf[int(self.u_idx), int(self.v_idx)]
        else:
            img_data = self.lf[int(self.u_idx), int(self.v_idx)]
        
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
        
        
def numpy_to_image_widget_value(data, format="png", quality=15):
    buffer = io.BytesIO()
    image = Image.fromarray(data)
    image.save(buffer, format= format, quality= quality)
    return buffer.getvalue()
    
        
