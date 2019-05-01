import numpy as np
from data.dataimport import import_data
import matplotlib.pyplot as plt

from encoders.baseencoder import AbstractEncoder


def calc_mean_stdev(encodings):
    """
    Calculates the mean and the standard deviation for each component of all SemVecs.
    
    encodings -- a numpy array with the SemVecs
    
    return -- a tupel of two numpy arrays one with the mean and one with the standard deviation for every component
    """
    
    mean = np.mean(encodings, axis=0)
    stdev = np.std(encodings, axis=0)
    
    return mean, stdev


def get_encodings(encoder, data):
    """
    Calculates the encodings of all the given expressions
    
    encoder -- the encoder object for encoding
    data -- a list with the expressions you want to encode
    
    return -- a numpy array with the encodings (the SemVecs)
    """
    
    encodings = []
    # iterate over the expressions and encode each
    for expression in data:
        encodings.append(encoder.get_encoding(expression))
    
    # transform the list to a numpy array
    encodings = np.array(encodings)
    
    return encodings


def take_all_expressions(data_filename):
    """
    Extracts all expressions from a data file (all equivalence classes, the original as well as the noise)
    
    data_file -- the path to the file with the data
    
    return a list with the expressions
    """
    
    data = import_data(data_filename)
    expressions = []
    
    # iterate over the equivalence classes in the file
    for eq_class, code in data.items():
        # add the original expression
        expressions.append(code['original'])
        
        # add also the noise expressions
        for noise_expression in code['noise']:
            expressions.append(noise_expression)
    
    return expressions


def load_encoder(encoder_filename):
    """
    Loads the encoder from file
    
    encoder_filename -- the path to the file of the encoder
    
    return -- an object of the encoder
    """
    
    return AbstractEncoder.load(encoder_filename)



def plot_mean_stdev(mean, stdev, output_filename):
    """
    Plots the mean and the standard deviation for all components of several SemVecs
    
    mean -- an array of the mean values for all the components
    stdev -- an array of the standard deviation for all the components
    output_filename -- the path for the plot
    """
    
    # an array with the numbers of the components as x-values
    x = np.arange(len(mean))
    
    # make a plot with the mean values as y-values and the stdev as errorbar
    plt.errorbar(x, mean, yerr = stdev, linestyle = '', marker = 'o')
    plt.xlabel('component')
    plt.ylabel('value')
    plt.savefig(output_filename)



# vorläufige main-funktion
if __name__ == "__main__":
    
    encoder_filename = 'rnnsupervisedencoder-boolean5.pkl'
    data_filename = 'expressions-synthetic/split/boolean5-testset.json.gz'
    output_filename = 'boolean5-test_all.svg'
    
    encoder = load_encoder(encoder_filename)
    expressions = take_all_expressions(data_filename)
    encodings = get_encodings(encoder, expressions)
    mean, stdev = calc_mean_stdev(encodings)
    plot_mean_stdev(mean, stdev, output_filename)
