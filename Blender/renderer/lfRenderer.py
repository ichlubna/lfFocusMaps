bl_info = {
    "name": "Light Field Asset Generator",
    "author": "ichlubna",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "3D View side panel",
    "description": "Generates a plane with material based on the input light field grid",
    "warning": "",
    "doc_url": "",
    "category": "Material"
}

import bpy
import os
import mathutils
import math
import numpy as np
from pathlib import Path

materialName="LFMaterial"
def setMaterialValues(context, nodeName, value):
    material = bpy.data.materials[materialName]
    for n in material.node_tree.nodes:
        if n.label == nodeName:
            n.outputs["Value"].default_value = value
            
def updateFocus(self, context):
    setMaterialValues(context, "Focus", context.scene.LFFocus)

def updateGridAspect(self, context):
    setMaterialValues(context, "Grid aspect", context.scene.LFGridAspect)
    
def updateQuilt(self, context, texture):
    group = bpy.data.node_groups["SamplingGroup"]
    for n in group.nodes:
        if n.label == "Texture":
            n.image = texture

class LFReader:
    cols = 0
    rows = 0
    files = []
    path = ""

    def loadDir(self, path):
        self.path = path
        files = sorted(os.listdir(path))
        length = Path(files[-1]).stem.split("_")
        if len(length) == 1:
            self.cols = len(files)
            self.rows = 1
        else:
            self.cols = int(length[1])+1
            self.rows = int(length[0])+1
        self.files = [files[i:i+self.cols] for i in range(0, len(files), self.cols)]

    def getColsRows(self):
        return [self.cols, self.rows]

    def getImagePath(self, row, col):
        filePath = os.path.join(self.path, self.files[row][col])
        return filePath

    def getImage(self, col, row):
        image = bpy.data.images.load(self.getImagePath(row,col), check_existing=True)
        return image
    
    def getResolution(self):
        image = bpy.data.images.load(self.getImagePath(0,0), check_existing=True)
        return image.size

class LFPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "LFGenerator"
    bl_label = "Takes input LF grid and creates plane with LF material"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(context.scene, "LFInput")
        col.prop(context.scene, "LFGridAspect")
        col.prop(context.scene, "LFViewMethod")
        if context.scene.LFViewMethod == "FIXED":
            col.prop(context.scene, "LFViewCoords")
        col.prop(context.scene, "LFSynthesisMethod")
        if context.scene.LFSynthesisMethod == "ONE":
            col.prop(context.scene, "LFFocus")
        else:
            col.prop(context.scene, "LFFocusRange")
        
        if  context.scene.LFInput == "":
            col.label(text="No input directory!")
        elif bpy.context.scene.camera == None:
            col.label(text="No active camera found!")
        else:
            col.operator("lf.generate", text="Generate")

class LFGenerator(bpy.types.Operator):
    """Generates the LF asset"""
    bl_idname = "lf.generate"
    bl_label = "Generate"
    
    def cameraView(self, context):
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
    
    def importMaterial(self, context):
        return
    
    def setMaterialAspectAndGridSize(self, context, aspect, gridSize):
        setMaterialValues(context, "Resolution aspect", aspect)
        setMaterialValues(context, "Grid size X", gridSize[0])
        setMaterialValues(context, "Grid size Y", gridSize[1])
    
    def createTexture(self, context):
        lf = LFReader()
        lf.loadDir(context.scene.LFInput)
        colsRows = lf.getColsRows()
        resolution = lf.getResolution()
        CHANNELS = 4
        
        self.setMaterialAspectAndGridSize(context, resolution[0]/resolution[1], colsRows)

        grid = bpy.data.images.new("Lightfield", width=colsRows[0]*resolution[0], height=colsRows[1]*resolution[1])
        gridArray = np.empty((colsRows[0]*resolution[0]*colsRows[1]*resolution[1]*CHANNELS), np.float32)
        imageSize = resolution[0]*resolution[1]*CHANNELS
        for col in range(colsRows[0]):
            for row in range(colsRows[1]):                
                image = lf.getImage(col, row)
                index = imageSize*((colsRows[1]-1-row)*colsRows[0]+col)
                gridArray[index:index+imageSize] = image.pixels
        result = gridArray.reshape((colsRows[1], colsRows[0], resolution[1], resolution[0]*CHANNELS))
        result = result.swapaxes(1,2)
        result = result.reshape(resolution[1]*colsRows[1], resolution[0]*colsRows[0]*CHANNELS)
        grid.pixels.foreach_set(result.ravel())
        grid.update()
        updateQuilt(self, context, grid)
        
        for col in range(colsRows[0]):
            for row in range(colsRows[1]):                
                image = lf.getImage(col, row)
                bpy.data.images.remove(image)
    
    def importMaterial(self, context):
        if not materialName in bpy.data.materials:
            with bpy.data.libraries.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "LFAssets.blend")) as (data_from, data_to):
                data_to.materials =[materialName]
    
    def createMaterial(self, context):
        self.importMaterial(context)
        self.createTexture(context)
    
    def createPlane(self, context):
        camera = context.scene.camera
        direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        direction = direction.normalized()
        position = camera.location+direction
        xSize = 2*math.tan(camera.data.angle_x*0.5)
        renderInfo = bpy.context.scene.render
        aspectRatio = renderInfo.resolution_y / renderInfo.resolution_x
        bpy.ops.mesh.primitive_plane_add(size=(xSize), location=position, rotation=camera.rotation_euler)
        context.object.dimensions[1] = xSize*aspectRatio
        context.object.data.materials.append(bpy.data.materials[materialName])

    def invoke(self, context, event):
        self.cameraView(context)
        self.createMaterial(context)
        self.createPlane(context)
        return {"FINISHED"}

def register():
    bpy.utils.register_class(LFGenerator)
    bpy.utils.register_class(LFPanel)
    bpy.types.Scene.LFInput = bpy.props.StringProperty(name="Input", subtype="FILE_PATH", description="The path to the input views in format cols_rows.ext", default="")
    bpy.types.Scene.LFViewMethod = bpy.props.EnumProperty(name="View", items=[("FREE", "Free", "The view changes according to the viewing angle in viewport.", 0), ("FIXED", "Fixed", "The view is fixed at the defined coordinates.", 1)])
    bpy.types.Scene.LFSynthesisMethod = bpy.props.EnumProperty(name="Focusing", items=[("ALL", "All-focused", "The scene is focused everywhere.", 0), ("ONE", "One-distance", "The focusing can be manually adjusted.", 1)])
    bpy.types.Scene.LFViewCoords = bpy.props.FloatVectorProperty(name="View coordinates", size=2, description="Normalized view coordinates", default=(0.5,0.5), min=0, max=1)
    bpy.types.Scene.LFFocusRange = bpy.props.FloatVectorProperty(name="Focus range", size=2, description="Starting and ending focusing range", default=(0.0,0.5))
    bpy.types.Scene.LFFocus = bpy.props.FloatProperty(name="Focus", description="The focusing distance.", update=updateFocus)
    bpy.types.Scene.LFGridAspect = bpy.props.FloatProperty(name="Grid ratio", description="The aspect ratio of the spacing between the capturing cameras.", default=1, update=updateGridAspect)
       
def unregister():
    bpy.utils.unregister_class(LFGenerator)
    bpy.utils.unregister_class(LFPanel)
    
if __name__ == "__main__" :
    register()        
