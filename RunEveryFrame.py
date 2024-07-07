import bpy
import random
import csv




def quantize(value, n):
    # Calculate the size of each part
    part_size = 1.0 / n
    # Find which part the value falls into
    part = int(value / part_size)
    # Calculate the center of the part
    center = (part * part_size) + (part_size / 2)
    # Ensure the center value does not exceed 1
    return min(center, 1)

def pre_frame(scene):
    
    path = scene.render.filepath
    
    print(scene.frame_current)    
    random.seed(scene.frame_current)
    
    mat = bpy.data.materials["DatasetNewTest"]
    currentTree = mat.node_tree
    
    nodes = currentTree.nodes    
    valuesNode = nodes["RandomValues"]
    valuesTree = valuesNode.node_tree
    
    outNode = valuesTree.nodes["Group Output"]
    
    values = [0 for i in range(51)]
    values[0] = scene.frame_current

    col_names = [0 for i in range(52)]
    col_names[0] = 'frame'
    
    
    for i, output in enumerate(outNode.inputs):
        
        if output.type != "VALUE":
            continue
        
        #print(output.name)
        value = round(random.uniform(0,1),3)
        
        nameSplit = output.name.split(" ")
        if nameSplit[0] == "Mult":
            value = quantize(value, int(nameSplit[1]))
        
        
        output.default_value = value
        values[i+1] = value
        col_names[i+1] = output.name
        
    strValues = [str(int) for int in values]
    bpy.context.scene.render.stamp_note_text = " ".join(strValues)
    
    with open(os.path.join(path, f"parameters_frame_{scene.frame_current}.txt", "w")) as file:
        file.write(", ".join(col_names))
        file.write(", ".join(strValues))
        
#    with open(path + ".csv", "a+", newline="") as file:        
#        writer = csv.writer(file)
#        writer.writerow(values)
        
    #print(values)

    
    

bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(pre_frame)
