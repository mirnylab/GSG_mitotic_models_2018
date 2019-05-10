import numpy as np


def make_helical_loopbrush(
        L,
        helix_radius,
        helix_step,
        loops,
        bb_linear_density=1.0,
        random_loop_orientations=False,
        bb_random_shift=0):
    '''
    Generate a conformation of a loop brush with a helically folded backbone.
    In this conformation, loops are folded in half and project radially
    from the backbone. 
    
    Parameters
    ----------
    L : int
        Number of particles.
    helix_radius: float
        Radius of the helical backbone.
    helix_step: float
        Axial step of the helical backbone.
    loops: a list of tuples [(int, int)]
        Particle indices of (start, end) of each loop.
    bb_linear_density: float
        The linear density of the backbone, 
        num_particles / unit of backbone length 
    random_loop_orientations: bool
        If True, then align loops at random angles, 
        otherwise align them along the radius set by
        the location of their base with respect to the 
        center of the helix.
    bb_random_shift : float
        Add a random shift along all three coordinates to the backbone.
        The default value is 0.

    Returns 
    -------
    coords: np.ndarray
        An Lx3 array of particle coordinates.
    
    '''
    coords = np.zeros(shape=(L,3))
    loopstarts = np.array([min(i) for i in loops])
    loopends = np.array([max(i) for i in loops])
    looplens = loopends - loopstarts

    if bool(loops):
        bbidxs = np.concatenate(
            [np.arange(0,loopstarts[0]+1)]
            + [np.arange(loopends[i],loopstarts[i+1]+1)
               for i in range(len(loops)-1)]
            + [np.arange(loopends[-1], L)])
    else:
        bbidxs = range(L)
    bb_len = len(bbidxs)

    helix_turn_len = np.sqrt(
        (2.0 * np.pi * helix_radius)**2 + helix_step**2)
    helix_total_winding = 2.0 * np.pi * (bb_len - 1) / bb_linear_density / helix_turn_len

    bb_phases = np.linspace(0, helix_total_winding, bb_len)

    coords[bbidxs] = np.vstack(
        [helix_radius * np.sin(bb_phases),
         helix_radius * np.cos(bb_phases),
         bb_phases / 2.0 / np.pi * helix_step]).T
    coords[bbidxs] += (
        np.random.random(bb_len * 3).reshape(bb_len, 3) * kwargs.get('bb_random_shift',0))

    for i in range(len(loops)):
        if random_loop_orientation:
            bb_u = coords[loops[i][1]] - coords[loops[i][0]]
            u = np.cross(bb_u, bb_u+(np.random.random(3)*0.2-0.1))
            u[2] = 0
        else:
            u = (coords[loops[i][0]] + coords[loops[i][1]])/2
            u[2] = 0

        u /= (u**2).sum()**0.5
        for j in range(looplens[i] // 2):
            coords[loopstarts[i]+j+1] = coords[loopstarts[i]+j] + u
            coords[loopends[i]-j-1]   = coords[loopends[i]-j]   + u

    return coords
