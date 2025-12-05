import numpy as np
import uaibot as ub

def check_free(_p, _pc, _radius, _bounds):
    
    if _p[0,0]< _bounds[0] or _p[0,0]>_bounds[1] or \
       _p[1,0]< _bounds[2] or _p[1,0]>_bounds[3] or \
       _p[2,0]< _bounds[4] or _p[2,0]>_bounds[5]:
           return False
    else:

        return _pc.projection(_p)[1]>1.2*_radius
    
def sample_random_direction(_p, _p_goal, _step_size):
    
    if np.random.uniform()<0.5:
        vec = np.random.randn(3)
        vec = vec/np.linalg.norm(vec)
        return _step_size* np.matrix(vec).T
    else:
        return _p_goal-_p

def is_line_free(_p1, _p2, _pc, _radius, _bounds):
    dist = np.linalg.norm(_p1-_p2)
    no_steps = int(dist/0.05)+1
    for i in range(no_steps):
        alpha = i/(no_steps-1+0.01)
        point = (1 - alpha) * _p1 + alpha * _p2
        if not check_free(point, _pc, _radius, _bounds):
            return False
    return True

def simplify_path(_path, _pc, _radius, _bounds):
    if len(_path) <= 2:
        return _path

    simplified = [_path[0]]
    i = 0
    while i < len(_path) - 1:
        j = len(_path) - 1
        while j > i + 1:
            if is_line_free(_path[i], _path[j], _pc, _radius, _bounds):
                break
            j -= 1
        simplified.append(_path[j])
        i = j

    return simplified

def random_path_planner(_p_start, _p_goal, _pc, _radius, _bounds,  _max_steps=16000, _step_size=0.5):
    path = [_p_start]
    current = _p_start.copy()

    found = False
    
    
    for i in range(_max_steps):
        direction = sample_random_direction(current, _p_goal, _step_size)
        new_point = current + direction 

        # Check line from current to new_point is free
        if not is_line_free(current, new_point, _pc, _radius, _bounds):
            continue

        path.append(new_point)
        
        current = new_point

        # Try to connect to goal if close enough
        if np.linalg.norm(_p_goal - current) < _step_size:
            if is_line_free(current, _p_goal, _pc, _radius, _bounds):
                path.append(_p_goal)
                found = True
                break

    if found:
        simpl_path = simplify_path(path, _pc, _radius, _bounds)
        print("Path found with "+str(len(simpl_path)-1))
        return simpl_path[1:]
    else:
        print("Path not found!")
        return [_p_goal]

def get_R_from_normal_vector(n):
    # n is the normal axis passed as arg
    n = np.asarray(n, dtype=float)
    z = n / np.linalg.norm(n)
    # Choose a helper vector
    if abs(z[2]) < 0.999:
        h = np.array([0, 0, 1])
    else:
        h = np.array([0, 1, 0])
    x = np.cross(h, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # Rotation matrix with z as the third column
    R = np.column_stack((x, y, z))
    # print(R)
    return R


## Not really know if we'll use

# Helper function to change drone ball color dynamically
def set_drone_ball_color(drone_index, color):
    """
    Change the color of a drone's ball.
    
    Parameters:
    -----------
    drone_index : int
        Index of the drone (0 to param_n_robots-1)
    color : str
        Color name (e.g., 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', etc.)
        or hex color code (e.g., '#FF0000')
    """
    if drone_index < 0 or drone_index >= len(drones):
        raise ValueError(f"Drone index {drone_index} is out of range [0, {len(drones)-1}]")
    
    # Access the ball object (it's the last object in the list)
    ball = drones[drone_index].list_of_objects[-1]
    
    # Change the color property (now supports direct assignment after Ball class modification)
    ball.color = color

def set_all_drone_colors(colors_list):
    """
    Change colors for all drones at once.
    
    Parameters:
    -----------
    colors_list : list
        List of color strings, one for each drone.
        If the list is shorter than the number of drones, it will cycle through colors.
    """
    for i in range(len(drones)):
        color = colors_list[i % len(colors_list)]
        set_drone_ball_color(i, color)

def hsv_to_hex(h, s=1.0, v=1.0):
    """
    Convert HSV color to hex string.
    
    Parameters:
    -----------
    h : float
        Hue value in range [0, 1] (0 = red, 1/3 = green, 2/3 = blue, 1 = red again)
    s : float
        Saturation in range [0, 1] (default: 1.0 for full saturation)
    v : float
        Value/brightness in range [0, 1] (default: 1.0 for full brightness)
    
    Returns:
    --------
    str : Hex color string in format '#RRGGBB'
    """
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convert to 0-255 range and format as hex
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return f'#{r:02x}{g:02x}{b:02x}'