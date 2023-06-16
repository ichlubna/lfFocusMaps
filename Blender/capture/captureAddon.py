bl_info = {
    "name": "Light Field Renderer",
    "author": "ichlubna",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "3D View side panel",
    "description": "Renders and stores views in a LF grid with the active camera in the center. Project render settings are used and all frames in the active range are rendered in case of animation.",
    "warning": "",
    "doc_url": "",
    "category": "Import-Export"
}

import bpy
import copy
import os
import time
import math
import numpy
import tempfile
import mathutils

class LFPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "LF"
    bl_label = "Light Field Renderer"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(context.scene, "LFAnalysis")
        if context.scene.LFAnalysis:
            col.prop(context.scene, "LFObjectOfInterest")  
            col.prop(context.scene, "LFAnalysisOverlap")
            col.prop(context.scene, "LFAnalysisSize")
            col.row().operator("lf.depth", text="Average depth")
            if context.scene.LFObjectOfInterest == None:
                col.row().label(text="Select an object")
            else:
                col.row().operator("lf.analyze", text="Analyze")
      
        col.prop(context.scene, "LFStep")
        col.prop(context.scene, "LFGridSize")
        col.prop(context.scene, "LFAnimation")
        col.row().operator("lf.render", text="Render")
        col.row().operator("lf.preview", text="Preview")
        if context.scene.camera == None:
            rowPreview.enabled = rowRender.enabled = False
        if context.scene.camera == None:
            col.label(text="No active camera set!")
        if context.scene.LFRunning:
            col.label(text="Running...")
            row = self.layout.row()
            row.prop(context.scene, "LFProgress")
            row.enabled = False

class CameraTrajectory():
    cameraVectors = None
    
    class CameraVectors():
        down = mathutils.Vector((0.0, 0.0, 0.0))
        direction = mathutils.Vector((0.0, 0.0, 0.0))
        right = mathutils.Vector((0.0, 0.0, 0.0))
        position = mathutils.Vector((0.0, 0.0, 0.0))
              
        def offsetPosition(self, step, coords, size):   
            centering = copy.deepcopy(coords)
            centering.x -= (size.x-1)/2.0
            centering.y -= (size.y-1)/2.0
            newPosition = copy.deepcopy(self.position)
            newPosition += self.down*step.y*centering.y
            newPosition += self.right*step.x*centering.x
            return newPosition
        
    def initCameraVectors(self, context):
        self.cameraVectors = self.getCameraVectors(context)     
    
    def getCameraVectors(self, context):
        vectors = self.CameraVectors()
        camera = context.scene.camera 
        vectors.position = mathutils.Vector(camera.location)
        vectors.down = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, -1.0, 0.0))
        vectors.down = vectors.down.normalized()
        vectors.direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        vectors.direction = vectors.direction.normalized()
        vectors.right = vectors.down.cross(vectors.direction)
        return vectors
    
    def getCurrentCoords(self, context):
        x = context.scene.LFCurrentView[0]
        y = context.scene.LFCurrentView[1]
        return x,y
    
    def currentPosition(self, context):
        x, y = self.getCurrentCoords(context)
        step = mathutils.Vector(context.scene.LFStep)
        gridSize = mathutils.Vector(context.scene.LFGridSize)
        return self.cameraVectors.offsetPosition(step, mathutils.Vector((x,y)), gridSize) 
    
    def createTrajectory(self, context):   
        camera = self.getCameraVectors()
        gridSize = mathutils.Vector(context.scene.LFGridSize)    
        trajectory = [[copy.deepcopy(camera.position) for x in range(int(gridSize.x))] for y in range(int(gridSize.y))]
        step = mathutils.Vector(context.scene.LFStep)
        for x in range(int(gridSize.x)):
            for y in range(int(gridSize.y)):
                trajectory[x][y] = camera.offsetPosition(step, mathutils.Vector((x,y)), gridSize)
        return trajectory

class CameraManager(CameraTrajectory):
    originalCamera = None
    
    def deselectAllButActiveCamera(self, context):
        for obj in bpy.context.selected_objects:
            obj.select_set(False) 
        camera = context.scene.camera
        camera.select_set(True)
    
    def duplicateCamera(self, context):
        self.initCameraVectors(context)
        self.deselectAllButActiveCamera(context)
        self.originalCamera = context.scene.camera 
        bpy.ops.object.duplicate(linked=False)
        self.originalCamera.select_set(False)
        newCamera = bpy.context.selected_objects[0]
        if newCamera.animation_data != None:
            newCamera.animation_data.action = None
        context.scene.camera = newCamera
        return newCamera  
    
    def restoreCamera(self, context):
        self.deselectAllButActiveCamera(context)
        bpy.ops.object.delete() 
        bpy.context.scene.camera = self.originalCamera
        
    def placeCamera(self, context):
        camera = context.scene.camera
        camera.location = self.currentPosition(context)

class Progress(bpy.types.Operator, CameraManager):
    """Manages the progress"""
    bl_idname = "wm.progress"
    bl_label = "Progress"
    timer = None
    timerPeriod = 0.65
    totalCount = 0
    totalCountInTime = 0
    frameCount = 1
    currentFrame = 0
        
    def initPadding(self, context):
        context.scene.LFIndexNamePadding[0] = len(str(context.scene.LFGridSize[0]))
        context.scene.LFIndexNamePadding[1] = len(str(context.scene.LFGridSize[0]))
        context.scene.LFIndexNamePadding[2] = len(str(context.scene.frame_end))
        
    def initVars(self, context): 
        context.scene.LFCurrentView[0] = 0
        context.scene.LFCurrentView[1] = 0
        context.scene.LFCurrentView[2] = 0
        self.totalCount = context.scene.LFGridSize[0]*context.scene.LFGridSize[1]
        self.totalCountInTime = copy.deepcopy(self.totalCount)      
        self.currentFrame = context.scene.frame_current 
        if context.scene.LFAnimation:
            self.frameCount = context.scene.frame_end - context.scene.frame_start + 1
            self.totalCountInTime *= self.frameCount
            self.currentFrame = context.scene.frame_start
        self.initPadding(context) 
    
    def reInitCamera(self, context):
        self.restoreCamera(context)
        self.duplicateCamera(context) 
    
    def updateFrame(self, context):
        if (context.scene.LFCurrentView[2] % self.totalCount) == 0:
            currentFrame = context.scene.LFCurrentView[2] // self.totalCount
            currentFrame += bpy.context.scene.frame_start
            context.scene.frame_set(currentFrame)
            if currentFrame != bpy.context.scene.frame_start:
                self.reInitCamera(context)
    
    def checkAndUpdateCurrentView(self, context):
        if context.scene.LFAnimation:
            self.updateFrame(context)   
        context.scene.LFCurrentView[2] += 1 
        if context.scene.LFCurrentView[2] >= self.totalCountInTime:
            return False
        linear = context.scene.LFCurrentView[2] % self.totalCount
        context.scene.LFCurrentView[0] = linear % context.scene.LFGridSize[0] 
        context.scene.LFCurrentView[1] = linear // context.scene.LFGridSize[0]
        return True    
    
    def updateProgress(self, context):
        context.scene.LFProgress = (context.scene.LFCurrentView[2]/(self.totalCountInTime))*100

    def modal(self, context, event):
        if event.type in {'ESC'} or context.scene.LFShouldEnd:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':            
            if context.scene.LFCurrentTaskFinished:
                self.placeCamera(context)
                if context.scene.LFIsPreview:
                    bpy.ops.lf.preview()
                else:
                    bpy.ops.lf.render()
                notFinished = self.checkAndUpdateCurrentView(context)
                if not notFinished:
                    self.deferredCancel(context)
                    return {'PASS_THROUGH'}
                self.updateProgress(context)

        return {'PASS_THROUGH'}

    def execute(self, context):
        context.scene.LFProgress = 0.0
        context.scene.LFRunning = True
        context.scene.LFShouldEnd = False
        self.initVars(context)
        if context.scene.LFAnimation:
            context.scene.frame_set(bpy.context.scene.frame_start)
        self.duplicateCamera(context) 
        wm = context.window_manager
        self.timer = wm.event_timer_add(self.timerPeriod, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def deferredCancel(self, context):
        context.scene.LFShouldEnd = True
        context.scene.LFProgress = 100.0

    def cancel(self, context):
        self.restoreCamera(context)
        context.scene.LFRunning = False
        wm = context.window_manager
        wm.event_timer_remove(self.timer)

class LFDepth(bpy.types.Operator):
    """ Computes average depth of the scene and creates an empty there """
    bl_idname = "lf.depth"
    bl_label = "Depth"
    
    class BackupData:
        resolution = [0,0]
        fileFormat = None
        overrideMaterial = None
        colorDepth = None
        colorMode = None
        world = None
        engine = None
        samples = 0
        
    backupData = BackupData()
    
    def pushBackup(self, context):
        renderInfo = bpy.context.scene.render
        self.backupData.resolution[0] = renderInfo.resolution_x
        self.backupData.resolution[1] = renderInfo.resolution_y
        self.backupData.fileFormat = renderInfo.image_settings.file_format 
        self.backupData.overrideMaterial = context.window.view_layer.material_override
        self.backupData.colorDepth = renderInfo.image_settings.color_depth
        self.backupData.colorMode = renderInfo.image_settings.color_mode  
        self.backupData.world = bpy.context.scene.world    
        self.backupData.engine = bpy.context.scene.render.engine
        self.backupData.samples = bpy.context.scene.cycles.samples
        
    def popBackup(self, context):
        renderInfo = bpy.context.scene.render
        renderInfo.resolution_x = self.backupData.resolution[0]
        renderInfo.resolution_y = self.backupData.resolution[1]
        renderInfo.image_settings.file_format = self.backupData.fileFormat
        context.window.view_layer.material_override = self.backupData.overrideMaterial
        renderInfo.image_settings.color_depth = self.backupData.colorDepth
        renderInfo.image_settings.color_mode = self.backupData.colorMode
        bpy.context.scene.world = self.backupData.world
        bpy.context.scene.render.engine = self.backupData.engine
        bpy.context.scene.cycles.samples = self.backupData.samples
    
    def createWorld(self, context):
        materialName = "LFworldMaterial"
        material = bpy.data.worlds.get(materialName) or bpy.data.worlds.new(name=materialName)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        
        materialOut = material.node_tree.nodes.new('ShaderNodeOutputWorld')
        materialOut.location[0]=400  
        background = material.node_tree.nodes.new('ShaderNodeBackground')
        background.location[0]=200 
        
        background.inputs[1].default_value = 0
        
        material.node_tree.links.new(materialOut.inputs[0], background.outputs[0])        
    
    def createMaterial(self, context):
        materialName = "LFcoordsMaterial"
        material = bpy.data.materials.get(materialName) or bpy.data.materials.new(name=materialName)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        
        materialOut = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        materialOut.location[0]=400  
        emission = material.node_tree.nodes.new('ShaderNodeEmission')
        emission.location[0]=200   
        divide = material.node_tree.nodes.new('ShaderNodeVectorMath')
        divide.operation="DIVIDE"
        divide.location[0]=0  
        planes = material.node_tree.nodes.new('ShaderNodeCombineXYZ')
        planes.location[0]=-200
        planes.location[1]=-200
        multiply = material.node_tree.nodes.new('ShaderNodeVectorMath')
        multiply.operation="MULTIPLY"
        multiply.location[0]=-200  
        cameraData = material.node_tree.nodes.new('ShaderNodeCameraData')
        cameraData.location[0]=-400

        clip = bpy.context.scene.camera.data.clip_end
        planes.inputs[0].default_value = clip
        planes.inputs[1].default_value = clip
        planes.inputs[2].default_value = clip

        material.node_tree.links.new(multiply.inputs[0], cameraData.outputs[0])
        material.node_tree.links.new(multiply.inputs[1], cameraData.outputs[2])
        material.node_tree.links.new(divide.inputs[0], multiply.outputs[0])
        material.node_tree.links.new(divide.inputs[1], planes.outputs[0])
        material.node_tree.links.new(emission.inputs[0], divide.outputs[0])
        material.node_tree.links.new(materialOut.inputs[0], emission.outputs[0])
        
        return material
    
    def setRenderSettings(self, context):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 16
        renderInfo = bpy.context.scene.render
        renderInfo.image_settings.file_format = "OPEN_EXR"
        renderInfo.image_settings.exr_codec = "ZIP"
        renderInfo.image_settings.color_depth = "32"
        renderInfo.image_settings.color_mode = "RGBA"
        rx = 1920
        ry = int(rx*(float(renderInfo.resolution_y)/renderInfo.resolution_x))
        renderInfo.resolution_x = rx
        renderInfo.resolution_y = ry
    
    def renderDepth(self, context):        
        self.pushBackup(context)
        
        self.setRenderSettings(context)
        
        material = self.createMaterial(context)
        context.window.view_layer.material_override = material
        bpy.context.scene.world = self.createWorld(context)
        
        bpy.ops.render.render( write_still=False )
        depth = 0  
        with tempfile.TemporaryDirectory() as tmpDir:   
            tmpFile = os.path.join(tmpDir,"coords.exr")     
            bpy.data.images['Render Result'].save_render(tmpFile)
            image = bpy.data.images.load(tmpFile)
            depths = numpy.array(image.pixels)[2::4]
            filter = depths > 0.00001
            depth = numpy.mean(depths[filter])          

        self.popBackup(context)
        if math.isnan(depth):
            depth = 0
        return depth*bpy.context.scene.camera.data.clip_end
    
    def createObject(self, context, depth):
         empty = None
         camera = context.scene.camera
         direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
         bpy.ops.object.empty_add(type='PLAIN_AXES', location = camera.location + direction.normalized()*depth) 
         bpy.context.active_object.name = 'LFAverageDepth' 
         return bpy.context.active_object
        
    def execute(self, context):
        depth = self.renderDepth(context)
        empty = self.createObject(context, depth)
        context.scene.LFObjectOfInterest = empty        
        return {"FINISHED"}

class LFAnalyze(bpy.types.Operator):
    """ Analyses the scene for optimal capturing """
    bl_idname = "lf.analyze"
    bl_label = "Analyze"
    
    def distanceToObject(self, context):
        object = context.scene.LFObjectOfInterest
        camera = context.scene.camera
        return (object.location-camera.location).length
    
    def getOffset(self, context):
        camera = context.scene.camera.data
        d = self.distanceToObject(context)
        o = context.scene.LFAnalysisOverlap
        fx=math.tan(camera.angle/2)
        renderInfo = bpy.context.scene.render
        aspect = renderInfo.resolution_x/renderInfo.resolution_y        
        fy=fx/aspect
    
        x = 2*d*fx-o
        y = 2*d*fy-o
        
        scale = context.scene.LFAnalysisSize
        return (x*scale, y*scale)
    
    def execute(self, context):
        context.scene.LFStep = self.getOffset(context)
        return {"FINISHED"}

class LFRender(bpy.types.Operator, CameraTrajectory):
    """ Renders the LF structure """
    bl_idname = "lf.render"
    bl_label = "Render"

    def execute(self, context):
        renderInfo = bpy.context.scene.render
        originalPath = copy.deepcopy(renderInfo.filepath)
 
        x, y = self.getCurrentCoords(context)
        frameID = bpy.context.scene.frame_current
        pad = context.scene.LFIndexNamePadding 
        filename = os.path.join(str(frameID).zfill(pad[2]), str(y).zfill(pad[1])+"_"+str(x).zfill(pad[0]))
        renderInfo.filepath = os.path.join(originalPath, filename)
        bpy.ops.render.render(write_still=True)  
        
        renderInfo.filepath = originalPath 
        return {"FINISHED"}

    def invoke(self, context, event):
        context.scene.LFIsPreview = False
        bpy.ops.wm.progress()
        return {"FINISHED"}
    
class LFPreview(bpy.types.Operator, CameraTrajectory):
    """ Animated the camera motion in the grid """
    bl_idname = "lf.preview"
    bl_label = "Render" 

    def execute(self, context):
        context.scene.LFCurrentTaskFinished = False    
        #time.sleep(0.5)
        context.scene.LFCurrentTaskFinished = True
        return {"FINISHED"}

    def invoke(self, context, event):
        #print(self.createTrajectory(context))
        context.scene.LFIsPreview = True
        bpy.ops.wm.progress()        
        return {"FINISHED"}

def register():
    bpy.utils.register_class(Progress)
    bpy.utils.register_class(LFPanel)
    bpy.utils.register_class(LFRender)
    bpy.utils.register_class(LFPreview)
    bpy.utils.register_class(LFAnalyze)
    bpy.utils.register_class(LFDepth)
    bpy.types.Scene.LFObjectOfInterest = bpy.props.PointerProperty(name="Object of interest", description="The distance of this object will be used in the offset calculation", type=bpy.types.Object)
    bpy.types.Scene.LFStep = bpy.props.FloatVectorProperty(name="Step", size=2, description="The distance between cameras", min=0, default=(1.0,1.0), subtype="XYZ")
    bpy.types.Scene.LFGridSize = bpy.props.IntVectorProperty(name="Grid size", size=2, description="The total number of the views", min=2, default=(8,8), subtype="XYZ")
    bpy.types.Scene.LFAnimation = bpy.props.BoolProperty(name="Render animation", description="Will render all active frames as animation", default=False)
    bpy.types.Scene.LFAnalysis = bpy.props.BoolProperty(name="Recalculate offset", description="Calculates the vertical camera spacing optimally", default=False)
    bpy.types.Scene.LFAnalysisOverlap = bpy.props.FloatProperty(name="Overlap", description="The amount of overlap between the cameras in both directions", min=0, default=0.1)
    bpy.types.Scene.LFAnalysisSize = bpy.props.FloatProperty(name="Grid size", description="The scaling of the resulting grid", min=0, default=1.0)
    bpy.types.Scene.LFProgress = bpy.props.FloatProperty(name="Progress", description="Progress bar", subtype="PERCENTAGE",soft_min=0, soft_max=100, default=0.0)
    bpy.types.Scene.LFCurrentView = bpy.props.IntVectorProperty(name="Current view", size=3, description="The currenty processed view - XY and third is linear ID", default=(0,0,0))
    bpy.types.Scene.LFCurrentTaskFinished = bpy.props.BoolProperty(name="Current view finished", description="Indicates that the current view was processed", default=True)
    bpy.types.Scene.LFShouldEnd = bpy.props.BoolProperty(name="Deffered ending", description="Is set to true in the last view to show last progress", default=False)
    bpy.types.Scene.LFRunning = bpy.props.BoolProperty(name="Running", description="Indicates that the rendering or previewing is in progress", default=False)
    bpy.types.Scene.LFIsPreview = bpy.props.BoolProperty(name="Is preview", description="Switch between preview and render", default=False)
    bpy.types.Scene.LFIndexNamePadding = bpy.props.IntVectorProperty(name="Zero padding", size=3, description="Padding of the output indexing in filenames", default=(1,1,1))
    
    
def unregister():
    bpy.utils.unregister_class(Progress)
    bpy.utils.unregister_class(LFPanel)
    bpy.utils.unregister_class(LFRender)
    bpy.utils.unregister_class(LFPreview)
    bpy.utils.unregister_class(LFAnalyze)
    bpy.utils.unregister_class(LFDepth)
    
if __name__ == "__main__" :
    register()        
