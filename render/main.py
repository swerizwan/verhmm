import bpy
import os
import numpy as np
import sys

# Function to set up Blender scene
def setup_scene(root_dir, filename):
    """
    Sets up the Blender scene with appropriate settings.

    Args:
        root_dir (str): Root directory of the project.
        filename (str): Name of the file being processed.

    Returns:
        tuple: A tuple containing the output directory and the path to the blendshape file.
    """
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'MATCAP'
    bpy.context.scene.display.render_aa = 'FXAA'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 768
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    cam = bpy.data.objects['Camera']
    cam.scale = [2, 2, 2]
    bpy.context.scene.camera = cam

    output_dir = os.path.join(root_dir, filename)
    blendshape_path = os.path.join(root_dir, filename + '.npy')
    return output_dir, blendshape_path

# Function to render images
def render_images(obj, model_bsList, output_dir, bs):
    """
    Renders images for each blendshape.

    Args:
        obj (bpy.types.Object): The object representing the face model.
        model_bsList (list): List of blendshape names.
        output_dir (str): Output directory for rendered images.
        bs (numpy.ndarray): Blendshape values for each frame.
    """
    for i, curr_bs in enumerate(bs):
        for j, value in enumerate(curr_bs):
            obj.data.shape_keys.key_blocks[model_bsList[j]].value = value
        bpy.context.scene.render.filepath = os.path.join(output_dir, '{}.png'.format(i))
        bpy.ops.render.render(write_still=True)

def main():
    filename = str(sys.argv[-1])
    root_dir = str(sys.argv[-2])

    model_bsList = [
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
        "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
        "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
        "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
        "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
        "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"
    ]

    obj = bpy.data.objects["face"]

    output_dir, blendshape_path = setup_scene(root_dir, filename)

    bs = np.load(blendshape_path)

    render_images(obj, model_bsList, output_dir, bs)

if __name__ == "__main__":
    main()
