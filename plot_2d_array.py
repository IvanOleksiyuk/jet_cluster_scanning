import numpy as np
import matplotlib.pyplot as plt
def plot_2d_array(DATA, TOP_DOWN):
    print(DATA)
    print(TOP_DOWN)
    if TOP_DOWN:
        data=np.flipud(DATA)
    else:
        data=DATA
    w=data.shape[0]
    h=data.shape[1]
    
    # Limits for the extent
    x_start = 0
    x_end = w
    y_start = 0
    y_end = h
    
    extent = [x_start, x_end, y_start, y_end]
    
    # The normal figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', cmap='rainbow')
    # Add the text to each element
    jump_x = (x_end - x_start) / (2.0 * w)
    jump_y = (y_end - y_start) / (2.0 * h)
    x_positions = np.linspace(start=x_start, stop=x_end, num=w, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=h, endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            if not np.isnan(label): # e.g. missing data or triangular matrices 
                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, "{:.3g}".format(label), color='black', ha='center', va='center')
    fig.colorbar(im)
    plt.show()