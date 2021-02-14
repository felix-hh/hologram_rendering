import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# datastructures
#from pytorch3d.structures import Meshes, Textures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate, Transform3d, so3_log_map, so3_exponential_map

from pytorch3d.renderer.cameras import SfMPerspectiveCameras

# rendering components
"""from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, Materials, TexturedSoftPhongShader
)"""

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, Materials
)

#from pytorch3d.renderer.mesh.texturing import interpolate_face_attributes
from pytorch3d.ops import interpolate_face_attributes

# for Delaunay triangulation
import scipy.spatial as sp

# use scipy for interpolation
from scipy.interpolate import interpn

class LightfieldShader(nn.Module):
    """
        LightfieldShader
        The batch dimension is the number of views to render
    """

    def __init__(self, cameras):
        super().__init__()
        self.cameras = cameras

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """"
        LightfieldShader just returns interpolated vertex coordinates for a plane
        """
        # get the faces, normals, and textures from the mesh
        faces = meshes.faces_packed()  # (F, 3)
        verts = meshes.verts_packed(); # (V, 3)
        faces_verts = verts[faces]        
        Nv, H_out, W_out, K = fragments.pix_to_face.shape; 

        # pixel_verts: (Nv, H, W, K=1, 3) -> (Nv, K=1, H, W, 3) -> (Nv*K=1, H, W, 3)
        K = 1;
        pixel_verts = interpolate_face_attributes(
            fragments.pix_to_face[:,:,:,0:K], fragments.bary_coords[:,:,:,0:K,:], faces_verts)
        pixel_verts = pixel_verts.permute(0, 3, 1, 2, 4).view(Nv * K, H_out, W_out, 3)

        return pixel_verts
    
class LightfieldViewerModel(nn.Module):
    def __init__(self, device, dtype, zsep=1, init_cam=None, Nv=311, Np=128, scale=1):

        ################################################################
        ################################################################
        super().__init__()

        ################################################################
        # input:
        #
        # Nv - the number of vertices across in the planar mesh
        # Np - the number of pixels in the rendererd images
        # zsep - the separation between the XY and UV planes {defualt: 1}
        # scale - the maximum XY scale of the vertices in the plane {defualt: -1,1}
        ################################################################        
        
        # create a regular grid of spatial coords for uniform mesh
        self.zsep = zsep;
        self.Np = Np;
        xp = np.linspace(-scale,scale,Nv) # grid is scaled -1,1
        Y,X = np.meshgrid(xp, xp);

        # create 2D vertex list
        init_verts2d = torch.cat((torch.tensor(X, dtype=dtype).view(Nv*Nv,1), 
                             torch.tensor(Y, dtype=dtype).view(Nv*Nv,1)), 1).to(device) # (1, Nv*Nv, 2)
        self.verts2D = init_verts2d
        #print(self.verts2D)

        # the xy plane is at z = 0;
        # the uv plane is at z = zsep (default);
        depthXY = torch.zeros(Nv*Nv,1).to(device)
        depthUV = zsep*torch.ones(Nv*Nv,1).to(device)

        # create the 3D vertex list from the 2D vertices and the depthmap
        verts3DXY = torch.cat((self.verts2D, depthXY), 1); # (views,N*N,3)
        verts3DUV = torch.cat((self.verts2D, depthUV), 1); # (views,N*N,3)
        
        # use scipy to help with Delaunay triangulation of the faces
        tri = sp.Delaunay(self.verts2D.cpu().numpy());
        self.faces = torch.tensor(tri.simplices, dtype=torch.int64).to(device); # (1, F, 3) - datatype must be integer for indices

        ################################################################
        # choose initial camera model 
        ################################################################
        
        if init_cam == None:
            self.views = 1;
            # Get the position of the camera based on the spherical angles
            R, T = look_at_view_transform(dist=2, elev=0, azim=0, device=device) # (views,3,3), (views,3) 
            init_cam = SfMPerspectiveCameras(device=device, R=R, T=T);
        self.cameras = init_cam                
                
        # the rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=Np, 
            blur_radius=np.log(1. / 1e-6 - 1.)*1e-6, 
            faces_per_pixel=1, 
            bin_size=0
        )
        
        ################################################################
        # Create a Meshes object for XY/UV planes.
        ################################################################

        self.meshesXY = Meshes(
            verts=verts3DXY[None,:,:],   
            faces=self.faces[None,:,:]
        )
        self.meshesUV = Meshes(
            verts=verts3DUV[None,:,:],   
            faces=self.faces[None,:,:] 
        )

        # extend the meshes for each view
        self.views = self.cameras.R.shape[0];
        self.meshesXY = self.meshesXY.extend(self.views)
        self.meshesUV = self.meshesUV.extend(self.views)

        ################################################################
        # create the renderer with a ReflectionMappingShader
        ################################################################        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=LightfieldShader(cameras=self.cameras)
        )
        
    ################################################################
    # Rendering function - just spits out XYUV coords
    ################################################################
    def forward(self):
        
        # render the vertex coordinates
        imageXY = self.renderer(meshes_world=self.meshesXY)
        imageUV = self.renderer(meshes_world=self.meshesUV)
        
        return imageXY, imageUV

    ################################################################
    # Rendering function 
    # input: 
    #    lightfield - numpy array # Nu x Nv x Nx x Ny x C
    #    u,v,x,y - 1D numpy arrays # {Nu x 1, Nv x 1, Nx x 1, Ny x 1}
    #         - four sets of 1D grid coordinates in u,v,x,y that the lightfield is sampled on 
    #
    # output:
    #
    #   renderer_views - numpy array # views x Np x Np x C
    ################################################################
    def forward(self, lightfield,u,v,x,y):
        
        Nu, Nv, Nx, Ny, C = lightfield.shape;
        if Nu != len(u) or Nv != len(v) or  Nx != len(x) or Ny != len(y):
            raise Exception("dimensions of u,v,x,y coordinates must match the input lightfield")
        
        points = (u,v,x,y);
        
        # render the vertex coordinates for the XY and UV planes
        imageXY = self.renderer(meshes_world=self.meshesXY)
        imageUV = self.renderer(meshes_world=self.meshesUV)

        # convert from 2-plane XY/UV to XY/tangent space
        imageUV = (imageUV - imageXY)/self.zsep # UV is ray tangent coords 

        new_x =  imageXY[:,:,:,0].cpu().numpy() # views x Np x Np
        new_y =  imageXY[:,:,:,1].cpu().numpy() # views x Np x Np
        new_u =  imageUV[:,:,:,0].cpu().numpy() # views x Np x Np
        new_v =  imageUV[:,:,:,1].cpu().numpy() # views x Np x Np
        ix = np.stack((-new_v,new_u,-new_y,new_x),3) # views x Np x Np x 4
#         print(ix.shape)

        renderedViews = np.zeros((self.views, self.Np, self.Np, C), dtype=lightfield.dtype)
        for c in range(C):
            vals = lightfield[:,:,:,:,c]
#             print(vals.shape)
            renderedViews[:,:,:,c] = interpn(points, vals, ix, method='linear', bounds_error=False)
        return np.clip(renderedViews, 0, 1)

    


