To upload to box, create a box_credentials.txt file. The first line should contain the CLIENT_ID, the second line should contain CLIENT_SECRET and the third line ACCESS_TOKEN - This can be obtained from the box api webpage after logging in with your account. 

registered_to_raw_lightfield.ipynb takes a folder of captured images in a zig zag pattern and produces a .npy file. The results are kept in the root folder. 

npy_raw_to_render_views.ipynb takes a npy file, produces image renderings with interpolated views and size as specified, then uploads the results to box. The results are also kept in the rendered_views folder.  

visualize_renderings takes sample images in a results_rendering folder and produces some figures in the figs/ folder with matplotlib.

old_renderer_benton_train_lightfield.ipynb is the old code that uses a different renderer but produces warped results. 

the .py files contain useful functions - lightfield_canvas.py and lightfield_utils.py are the main ones used in the first two notebooks. 
