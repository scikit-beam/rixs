

def extract_regions(data, regions):
    '''Extracts a series of regions from a series of 2D images.

    This function takes in a list of 2D numpy arrays and a dictionary defining
    a group of regions. It outputs a dictionary containing each of the regions
    names as a keyword and a list of numpy arrays as the value.
        The input dictionary has the form:
        dict = { 'region1_name' : [left1, right1, top1, bottom1],
                 'region2_name' : [left2, right2, top2, bottom2],....}

    Parameters
    ----------
    data : list
        This is a list of 2D numpy arrays.
    regions : dict
        A dictionary of regions that the images in 'data' are to be
        extracted too.

    Returns
    -------
    out_dict : dict
        This is a  dictionary out_dict with the form:
            out_dict = {'region1_name': region1, .....}
        with an entry for each region in the dictionary regions and a
        corresponding numpy array.
    '''

    out_dict = {}
    for region in regions:
        out_dict[region] = []
        for arr in data:
            out_dict[region].append(arr[regions[region][0]:regions[region][1],
                                    regions[region][2]:regions[region][3]])
    return out_dict
