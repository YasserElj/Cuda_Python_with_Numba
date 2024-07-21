# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    
    nbins = histogram_out.shape[0]
    
    bin_width = (xmax-xmin)/nbins
    
    start = cuda.grid(1)
    
    step = cuda.gridsize(1)
    
    for i in range(start, x.shape[0], step):
        bin_number = (x[i] - xmin)//bin_width
        
        if bin_number >= 0 and bin_number < nbins:    
            cuda.atomic.add(histogram_out, bin_number, 1)    