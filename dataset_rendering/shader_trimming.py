import math
import pdb
from time import sleep

import bpy

mat = bpy.data.materials["DatasetNewTest"]
treeStack = [mat.node_tree]
grp_node_stack = []


def currentTree():
    return treeStack[-1]


def parentTree():
    if (len(treeStack) > 1):
        return treeStack[-2]


def clamp_float(val, min_=0, max_=1):
    return min(max(val, min_), max_)


print("-------------------------")


def backTrack(node):

    node.use_custom_color = True
    node.color = (0, 0, 1)

    # input('Press Enter to continue: ')

    for input_pin in node.inputs:
        if len(input_pin.links) > 0:
            link = input_pin.links[0]
            optimization = backTrack(link.from_node)
            if optimization is not None:
                currentTree().links.remove(link)
                setDefaultValue(input_pin, optimization)

    # node.use_custom_color = False

    # if node.name == 'Group.007' or 'Hard Switch 8' in node.name:
    #     print(node.name)
    #     raise ValueError()

    optim = optimizeNode(node)
    node.color = (node.color[0] / 2, node.color[1] / 2, node.color[2] / 2)

    return optim


# this is not great at all. AT ALL there is more stuff not managed then there is stuff managed.
# seams to work with try catch. its so disgusting though i might wanna.. nvm
def setDefaultValue(inputSocket, value):
    try:
        inputSocket.default_value = value

    except:
        try:
            print(value)
            inputSocket.default_value = [value, value, value]
        except TypeError:
            inputSocket.default_value = value[0] * 0.2126 + value[1] * 0.7152 + value[2] * 0.0722


def cleanUpCorpses(tree):
    nodesToRemove = []

    for node in tree.nodes:
        hasOut = False
        # print(node)

        if node.type == "OUTPUT_MATERIAL" or node.type == "GROUP_OUTPUT" or node.type == "GROUP_INPUT":
            continue

        for out in node.outputs:
            if len(out.links) > 0:
                hasOut = True
                break
        if hasOut:
            continue
        else:
            nodesToRemove.append(node)

    # print (nodesToRemove)

    if len(nodesToRemove) == 0:
        return 0

    for node in nodesToRemove:
        tree.nodes.remove(node)

    return cleanUpCorpses(tree)


def num_conected_inputs(node):
    connected = 0

    for ins in node.inputs:
        if len(ins.links) > 0:
            connected += 1

    return connected


def optimizeNode(node):
    type = node.type

    if type in optimizeable:
        node.color = (0, 1, 0)
        return optimizeable[type](node)
    else:
        node.color = (1, 0, 0)
        print(type + " not defined yet")

    return None


def optimizeMixRGB(node):
    return None


def optimizeMixShader(node):
    return None


def optimizeCombXYZ(node):
    if num_conected_inputs(node):
        return None

    return (node.inputs[0].default_value, node.inputs[1].default_value, node.inputs[2].default_value)


def optimizeCombColor(node):
    if num_conected_inputs(node):
        return None

    return (node.inputs[0].default_value, node.inputs[1].default_value, node.inputs[2].default_value, 1)


def optimizeMapRange(node):
    if num_conected_inputs(node):
        return None
    x = node.inputs[0].default_value
    a = node.inputs[1].default_value
    b = node.inputs[2].default_value
    c = node.inputs[3].default_value
    d = node.inputs[4].default_value

    y = (x - a) / (b - a) * (d - c) + c
    return y


def optimise_clamp(node):
    if num_conected_inputs(node):
        return None
    val = node.inputs[0].default_value
    min = node.inputs[1].default_value
    max = node.inputs[2].default_value
    if val < min:
        return min
    if val > max:
        return max
    return val


#############################################################################


def optimizeMath(node):
    if node.operation in optimizeMathOp:
        return optimizeMathOp[node.operation](node)
    else:
        print(node.operation + " not defined yet")
        return None


def optimizeAdd(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0]
        in1 = node.inputs[1]
        res = in0.default_value + in1.default_value
        if node.use_clamp:
            res = clamp_float(res)
        return res

    connected_input = None
    other_zero = False

    for i, input_socket in enumerate(node.inputs):
        if input_socket.is_linked:
            connected_input = input_socket
        else:
            if input_socket.default_value == 0.0:
                other_zero = True

    if other_zero and not node.use_clamp:
        for link in node.outputs[0].links:
            currentTree().links.new(connected_input.links[0].from_socket, link.to_socket)


def optimizeMul(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0].default_value
        in1 = node.inputs[1].default_value
        res = in0 * in1
        if node.use_clamp:
            res = clamp_float(res)
        return res

    connected_input = None
    other_one = False

    for i, input_socket in enumerate(node.inputs):
        if input_socket.is_unavailable:
            continue
        if input_socket.is_linked:
            connected_input = input_socket
        else:
            if input_socket.default_value == 1.0:
                other_one = True

            elif input_socket.default_value == 0.0:
                node.label = f'{node.inputs[0].default_value} {node.inputs[1].default_value} returned 0'
                return 0.0

    if other_one and not node.use_clamp:
        for link in node.outputs[0].links:
            node.label = 'jumped'
            currentTree().links.new(connected_input.links[0].from_socket, link.to_socket)
            return None


def optimizeDiv(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0].default_value
        in1 = node.inputs[1].default_value
        res = in0 / in1
        if node.use_clamp:
            res = clamp_float(res)
        return res

    if node.inputs[0].is_linked and node.inputs[1].default_value == 1.0 and not node.use_clamp:
        for link in node.outputs[0].links:
            currentTree().links.new(node.inputs[0].links[0].from_socket, link.to_socket)
            return None
    elif not node.inputs[0].is_linked and node.inputs[0].default_value == 0.0:
        return 0.0

    return None


def optimizeSub(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0].default_value
        in1 = node.inputs[1].default_value
        res = in0 - in1
        if node.use_clamp:
            res = clamp_float(res)
        return res

    # other then with add, the order makes a difference now. only when the second input is zero we can kill this node
    if node.inputs[0].is_linked and node.inputs[1].default_value == 0.0 and not node.use_clamp:
        for link in node.outputs[0].links:
            currentTree().links.new(node.inputs[0].links[0].from_socket, link.to_socket)

    return None


def optimizeCom(node):
    if num_conected_inputs(node):
        return None
    val0 = node.inputs[0].default_value
    val1 = node.inputs[1].default_value
    val2 = node.inputs[2].default_value
    return abs(val1 - val0) < val2


def optimise_math_floor(node):
    if num_conected_inputs(node):
        return None
    res = math.floor(node.inputs[0].default_value)
    if node.use_clamp:
        return clamp_float(res)
    return res


def optimise_math_max(node):
    if num_conected_inputs(node):
        return None

    res = max(node.inputs[0].default_value, node.inputs[1].default_value)
    if node.use_clamp:
        return clamp_float(res)
    return res


def optimise_math_min(node):
    if num_conected_inputs(node):
        return None

    res = min(node.inputs[0].default_value, node.inputs[1].default_value)
    if node.use_clamp:
        return clamp_float(res)
    return res


def optimise_math_pow(node):
    num_connected = num_conected_inputs(node)
    if num_connected >= 2:
        return None

    if num_connected == 0:
        res = node.inputs[0].default_value**node.inputs[1].default_value
        if node.use_clamp:
            res = clamp_float(res)
        return res

    if node.inputs[0].is_linked and node.inputs[1].default_value == 1.0 and not node.use_clamp:
        for link in node.outputs[0].links:
            currentTree().links.new(node.inputs[0].links[0].from_socket, link.to_socket)
            return None

    elif node.inputs[0].is_linked and node.inputs[1].default_value == 0.0:
        return 1.0


def optimise_math_gt(node):
    if num_conected_inputs(node):
        return None

    res = node.inputs[0].default_value > node.inputs[1].default_value
    if node.use_clamp:
        return clamp_float(res)
    return res


#############################################################################


def optimizeVectMath(node):
    if num_conected_inputs(node):
        return None
    if node.operation in optimizeVecMathOp:
        return optimizeVecMathOp[node.operation](node)
    else:
        print(node.operation + " not defined yet")
        return None


def optimizeVecScale(node):
    if num_conected_inputs(node):
        return None

    vecXIn = node.inputs[0]
    vecYIn = node.inputs[1]
    vecZIn = node.inputs[2]
    scaleIn = node.inputs[3]

    return (vecXIn * scaleIn, vecYIn * scaleIn, vecZIn * scaleIn)


def optimise_vector_math_add(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0]
        in1 = node.inputs[1]
        print('vec add node has no inputs connected, values can be added and node removed')
        return (in0.default_value[0] + in1.default_value[0], in0.default_value[1] + in1.default_value[1],
                in0.default_value[2] + in1.default_value[2])

    connected_input = None
    other_zero = False

    for i, input in enumerate(node.inputs):
        if len(input.links) > 0:
            connected_input = input
        else:
            if input.default_value == (0.0, 0.0, 0.0):
                other_zero = True

    if other_zero:
        for link in node.outputs[0].links:
            currentTree().links.new(connected_input.links[0].from_socket, link.to_socket)
            print('vec add node has only one input and zero other input, can get removed')

    return None


def optimizeVecSub(node):
    num_inputs_connected = num_conected_inputs(node)

    if num_inputs_connected >= 2:
        return None

    if num_inputs_connected == 0:
        in0 = node.inputs[0]
        in1 = node.inputs[1]
        print('vec sub node has no inputs connected, values can be subtracted and node removed')
        return (in0.default_value[0] - in1.default_value[0], in0.default_value[1] - in1.default_value[1],
                in0.default_value[2] - in1.default_value[2])

    # other then with add, the order makes a difference now. only when the second input is zero we can kill this node
    if len(node.inputs[0].links) > 0 & node.inputs[1].default_value == (0.0, 0.0, 0.0):
        for link in node.outputs[0].links:
            currentTree().links.new(node.inputs[0].links[0].from_socket, link.to_socket)

    return None


#############################################################################


def optimizeGroup(node):
    input_not_connected_values = {}

    new_tree = node.node_tree.copy()
    node.node_tree = new_tree

    for i, input in enumerate(node.inputs):
        if not input.is_linked:
            if not hasattr(input, 'default_value'):
                continue
            input_not_connected_values[i] = input.default_value

    inputNode = new_tree.nodes["Group Input"]

    for i, output in enumerate(inputNode.outputs):
        if i not in input_not_connected_values:
            continue

        out_links = list(output.links)
        for link in out_links:
            target_socket = link.to_socket
            new_tree.links.remove(link)
            target_socket.default_value = input_not_connected_values[i]

    grp_node_stack.append(node)
    treeStack.append(new_tree)

    returnVal = backTrack(new_tree.nodes["Group Output"])
    cleanUpCorpses(currentTree())
    treeStack.pop()
    grp_node_stack.pop()

    # ungroup_group(node)

    return returnVal


def ungroup_group(node):
    sleep(0.1)
    currentTree().nodes.active = node
    node.select = True
    win = bpy.context.window
    scr = win.screen
    areas = [area for area in scr.areas if area.type == 'NODE_EDITOR']
    if areas:
        areas[0].spaces.active.node_tree = currentTree()
        regions = [region for region in areas[0].regions if region.type == 'WINDOW']
        if regions:
            try:
                with bpy.context.temp_override(window=win, area=areas[0], region=regions[0], screen=scr):
                    bpy.ops.node.group_ungroup('INVOKE_DEFAULT')
            except RuntimeError as e:
                print(f"Failed to ungroup node: {e}")


def ungroup_all_groups(tree):
    for node in tree.nodes:
        if node.type == 'GROUP':
            ungroup_all_groups(node.node_tree)

        ungroup_group(node)


def cleanUpEmptyInputs(currentTree):
    # inputNode = currentTree.nodes["Group Input"]
    # return
    print('cleanuptree')
    print(currentTree.name)

    interfaces = [interface for interface in currentTree.interface.items_tree if interface.in_out == 'INPUT']
    # print(interfaces)

    is_connected = set()

    for node in currentTree.nodes:
        if node.type != 'GROUP_INPUT':
            continue

        for i, socket in enumerate(node.outputs):

            if socket.is_linked:
                is_connected.add(i)

    not_connected_indices = set(range(len(interfaces))) - is_connected
    print(not_connected_indices)
    for i in sorted(list(not_connected_indices), reverse=True):
        currentTree.interface.remove(interfaces[i])
        # raise ValueError()

    # print(is_connected)
    #
    # for i in reversed(range(len(inputNode.outputs) - 1)):
    #     if len(inputNode.outputs[i].links) == 0:
    #         print(dir(inputNode.outputs[i]))
    #         currentTree.inputs.remove(currentTree.inputs[i])


def optimizeGrpOut(node):
    for i, ins in enumerate(node.inputs):
        if i == len(node.inputs) - 1:
            break
        if len(ins.links) == 0:
            # print(ins.name)
            # pdb.set_trace()
            for link in grp_node_stack[-1].outputs[i].links:
                link.to_socket.default_value = ins.default_value
                parentTree().links.remove(link)


#    if len(node.inputs) == 2:
#        if len(node.inputs[0].links) == 0:
#            return node.inputs[0].default_value
#    return None

#############################################################################


def optimizeValue(node):
    return node.outputs[0].default_value


def optimizeRerout(node):
    # print(node.name)
    if len(node.inputs[0].links) == 0:
        return node.inputs[0].default_value

    for link in node.outputs[0].links:
        currentTree().links.new(node.inputs[0].links[0].from_socket, link.to_socket)

    return None


#############################################################################

from collections import OrderedDict
from itertools import repeat


class values():
    average_y = 0
    x_last = 0
    margin_x = 300
    mat_name = ""
    margin_y = 250


def arrange_node_tree(ntree):

    # first arrange nodegroups
    n_groups = []
    for i in ntree.nodes:
        if i.type == 'GROUP':
            n_groups.append(i)

    while n_groups:
        j = n_groups.pop(0)
        nodes_iterate(j.node_tree)
        for i in j.node_tree.nodes:
            if i.type == 'GROUP':
                n_groups.append(i)

    nodes_iterate(ntree)


def nodes_odd(ntree, nodelist):
    nodes = ntree.nodes
    for i in nodes:
        i.select = False
    a = [x for x in nodes if x not in nodelist]
    for i in a:
        i.select = True


def outputnode_search(ntree):
    outputnodes = []
    for node in ntree.nodes:
        if not node.outputs:
            for input in node.inputs:
                if input.is_linked:
                    outputnodes.append(node)
                    break
    if not outputnodes:
        print("No output node found")
        return None
    return outputnodes


def nodes_iterate(ntree, arrange=True):
    nodeoutput = outputnode_search(ntree)
    if nodeoutput is None:
        return None
    a = []
    a.append([])
    for i in nodeoutput:
        a[0].append(i)
    level = 0
    while a[level]:
        a.append([])
        for node in a[level]:
            inputlist = [i for i in node.inputs if i.is_linked]
            if inputlist:
                for input in inputlist:
                    for nlinks in input.links:
                        node1 = nlinks.from_node
                        a[level + 1].append(node1)
            else:
                pass
        level += 1
    del a[level]
    level -= 1
    for x, nodes in enumerate(a):
        a[x] = list(OrderedDict(zip(a[x], repeat(None))))
    top = level
    for row1 in range(top, 1, -1):
        for col1 in a[row1]:
            for row2 in range(row1 - 1, 0, -1):
                for col2 in a[row2]:
                    if col1 == col2:
                        a[row2].remove(col2)
                        break
    if not arrange:
        nodelist = [j for i in a for j in i]
        nodes_odd(ntree, nodelist=nodelist)
        return None
    levelmax = level + 1
    level = 0
    values.x_last = 0
    while level < levelmax:
        values.average_y = 0
        nodes = [x for x in a[level]]
        nodes_arrange(ntree, nodes, level)
        level = level + 1
    return None


def nodes_arrange(ntree, nodelist, level):
    parents = []
    for node in nodelist:
        parents.append(node.parent)
        node.parent = None
        ntree.nodes.update()
    widthmax = max([x.dimensions.x for x in nodelist])
    xpos = values.x_last - (widthmax + values.margin_x) if level != 0 else 0
    values.x_last = xpos
    x = 0
    y = 0
    for node in nodelist:
        if node.hide:
            hidey = (node.dimensions.y / 2) - 8
            y = y - hidey
        else:
            hidey = 0
        node.location.y = y
        y = y - values.margin_y - node.dimensions.y + hidey
        node.location.x = xpos  # if node.type != "FRAME" else xpos + 1200
    y = y + values.margin_y
    center = (0 + y) / 2
    values.average_y = center - values.average_y
    for i, node in enumerate(nodelist):
        node.parent = parents[i]


#############################################################################


def void(node):
    return None


optimizeable = {
    "MIX_SHADER": optimizeMixShader,
    "MIX_RGB": optimizeMixRGB,
    "MATH": optimizeMath,
    "VALUE": optimizeValue,
    "REROUTE": optimizeRerout,
    "VECT_MATH": optimizeVectMath,
    "GROUP": optimizeGroup,
    "GROUP_OUTPUT": optimizeGrpOut,
    "COMBINE_COLOR": optimizeCombColor,
    "MAP_RANGE": optimizeMapRange,
    "COMBXYZ": optimizeCombXYZ,
    "COMBRGB": optimizeCombColor,
    "GROUP_INPUT": void,
    "TEX_NOISE": void,
    "TEX_VORONOI": void,
    "CLAMP": optimise_clamp,
}

optimizeMathOp = {
    "ADD": optimizeAdd,
    "MULTIPLY": optimizeMul,
    "DIVIDE": optimizeDiv,
    "SUBTRACT": optimizeSub,
    "COMPARE": optimizeCom,
    "FLOOR": optimise_math_floor,
    "MAXIMUM": optimise_math_max,
    "MINIMUM": optimise_math_min,
    "POWER": optimise_math_pow,
    "GREATER_THAN": optimise_math_gt,
}

optimizeVecMathOp = {
    "ADD": optimise_vector_math_add,
    "SCALE": optimizeVecScale,
    # "MULTIPLY":optimizeVecMul,
    # "DIVIDE":optimizeVecDiv,
    "SUBTRACT": optimizeVecSub
    # "COMPARE":optimizeVecCom,
}

nodes = currentTree().nodes
root = nodes["Material Output"]

# ungroup_all_groups(currentTree())
backTrack(root)
ungroup_all_groups(currentTree())
cleanUpCorpses(currentTree())
arrange_node_tree(currentTree())
