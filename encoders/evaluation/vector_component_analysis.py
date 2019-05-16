import numpy as np
from data.dataimport import import_data
import matplotlib.pyplot as plt
import os
import re

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


def take_expressions_eq_classes(data_filename):
    """
    Extracts all expressions sorted by equivalence classes and the equivalence classes from a data file.
    
    data_file -- the path to the file with the data
    
    return -- a tupel of two lists, the first one contains the eqiuvalence classes, the second one contains for each equivalence class a list with the expressions
    """
    
    data = import_data(data_filename)
    # a list for the equivalence classes
    eq_classes = []
    # a list of lists for the expressions
    expressionss = []
    
    # iterate over the equivalence classes in the file
    for eq_class, code in data.items():
        eq_classes.append(eq_class)
        
        # a list for the expressions
        expressions = []
        # add the original expression
        expressions.append(code['original'])
        
        # add also the noise expressions
        for noise_expression in code['noise']:
            expressions.append(noise_expression)
        
        # add the list of expressions to the list of lists
        expressionss.append(expressions)
    
    return eq_classes, expressionss


def symbol_is_in_expression(expression, symbol):
    """
    return if given symbol is present in expression
    """
    return not (re.search(symbol, expression) == None)


def symbol_is_only_symbol(expression, symbol, all_symbols):
    """
    return if given symbol is exclusive in expression
    """
    if(symbol_is_in_expression(expression, symbol)):
        for i in range(len(all_symbols)):
            if (symbol != all_symbols[i] and symbol_is_in_expression(expression, all_symbols[i])):
                #other symbol in expression
                return False
        #symbol is only symbol in expression
        return True
    else:
        #symbol is not in expression
        return False


def take_expressions_symbol(data_filename, symbol, all_symbols, exclusive = True):
    """
    Extracts all expressions for a given symbol from a data file
    (all equivalence classes, the original as well as the noise)

    data_file -- the path to the file with the data
    symbol -- symbol in expression
    all_symbols -- all possible symbols for file
    exclusive -- if True: the symbol has to be exclusive to the expression

    return a list with the expressions
    """

    data = import_data(data_filename)
    expressions = []

    # iterate over the equivalence classes in the file
    for eq_class, code in data.items():
        if(exclusive):
            inExpression = symbol_is_only_symbol(eq_class, symbol, all_symbols)
        else:
            inExpression = symbol_is_in_expression(eq_class, symbol)
        if(inExpression):
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


def find_min_max(arr, minimum, num):
    """
    finds the smallest or largest values of an array
    
    arr - the array with the values
    minimum - True, if you search for a minimum (False for maximum)
    num - the number of values you are searching for
    """
    
    # an array of the indices that sort the array in ascendin order
    idx = np.argsort(arr)
    
    if minimum:
        return idx[0:num]
    else:
        length = len(idx)
        return idx[length - num: length]


def plot_mean_stdev(mean, stdev, output_filename):
    """
    Plots the mean and the standard deviation for all components of several SemVecs
    
    mean -- an array of the mean values for all the components
    stdev -- an array of the standard deviation for all the components
    output_filename -- the path for the plot
    """
    
    # create a new figure for this plot
    plt.figure('new_figure')
    
    # an array with the numbers of the components as x-values
    x = np.arange(len(mean))
    
    # make a plot with the mean values as y-values and the stdev as errorbar
    plt.clf()
    plt.errorbar(x, mean, yerr = stdev, linestyle = '', marker = 'o')
    plt.xlabel('component')
    plt.ylabel('value')
    plt.savefig(output_filename)
    
    # close the figure
    plt.close('new_figure')


def plot_mean_all_in_one(mean, figure_name):
    """
    Plots the mean for all components of several SemVecs, every call of this method plots one line in the given figure
    
    mean -- an array of the mean values for all the components
    figure_name -- the name of the figure where you want to have a plot of the given means
    """
    
    # switch to the given figure
    plt.figure(figure_name)
    
    # an array with the numbers of the components as x-values
    x = np.arange(len(mean))
    
    plt.plot(x, mean, linestyle = '-', linewidth = 0.1, marker = '')


def save_plot_all_in_one(figure_name, output_filename):
    """
    Saves the all in one figure.
    
    figure_name -- the name of the figure you want to save
    output_filename -- the path for the plot
    """
    
    # switch to the given figure
    plt.figure(figure_name)
    
    # set labels and save the figure
    plt.xlabel('component')
    plt.ylabel('value')
    plt.savefig(output_filename)
    
    # close the figure
    plt.close(figure_name)


def plot_for_all_eqClasses_average(encoder, data_filename, dataset, path_to_output_file):
    """
    Save a plot for all equivalence classes for given dataset (average)
    Name of file: dataset-test_all.svg

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    dataset --  name of the used dataser
    path_to_output_file -- path for the output file
    """
    output_filename = path_to_output_file +  dataset + '-test_all.svg'
    expressions = take_all_expressions(data_filename)
    encodings = get_encodings(encoder, expressions)
    mean, stdev = calc_mean_stdev(encodings)
    plot_mean_stdev(mean, stdev, output_filename)


def plot_for_every_eqClass(encoder,  data_filename, dataset, path_to_output_file):
    """
    Save a plot for every equivalence class for given dataset
    Name of files: dataset-<nof expressions in equivalence class>-<name of equivalence class>.svg
    It also creates an 'all in one' plot, whre a line for every equivalence class is plottet to one figure
    Name of file: dataset-all_in_one.svg

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    dataset --  name of the used dataser
    path_to_output_file -- path for the output file
    """
    # set a name for the figure of the all in one plot
    all_in_one_name = 'all_in_one'
    
    eqClasses, expressionsByEqClass = take_expressions_eq_classes(data_filename)
    for i in range(len(eqClasses)):
        print("generating plot for:", eqClasses[i])
        output_filename = path_to_output_file +  dataset + '-' + str(len(expressionsByEqClass[i])) + '-' + eqClasses[i] + '.svg'
        encodings = get_encodings(encoder, expressionsByEqClass[i])
        mean, stdev = calc_mean_stdev(encodings)
        if len(expressionsByEqClass[i]) > 1:
            print(str(find_min_max(stdev, False, 4)))
        if len(expressionsByEqClass[i]) > 50:
            print(str((stdev > 0.05).sum()))
        # make the normal plot
        plot_mean_stdev(mean, stdev, output_filename)
        # plot the data of this iteration to the all in one plot
        plot_mean_all_in_one(mean, all_in_one_name)
    
    # create a name for the all in one plot and save it
    all_in_one_filename = path_to_output_file + dataset + '-all_in_one.svg'
    save_plot_all_in_one(all_in_one_name, all_in_one_filename)


def determine_regex(expression):
    """
    calculates the regular expression for given symbol
    """
    if(expression == "And"):
        return "And"
    elif(expression == "Or"):
        return "Or"
    elif(expression == "Not"):
        return "Not"
    elif(expression == "*"):
        return "[a-z0-9][*][a-z0-9]"
    elif(expression == "-"):
        return "[a-z0-9] [-] [a-z0-9]"
    elif(expression == "+"):
        return "[a-z0-9] [+] [a-z0-9]"
    elif(expression == "**"):
        return "[a-z][*][*][0-9]"


def plot_by_symbol(encoder, data_filename, dataset, path_to_output_file, all_symbols, exclusive = True):
    """
    Save a plot for every symbol in given dataset
    Name of files: dataset-<nof expressions used>-<name of symbol>.svg

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    dataset --  name of the used dataser
    path_to_output_file -- path for the output file
    all_symbols -- all used symbols in the file
    exclusive -- bool if symbol has to be exclusive in expression
    """

    #calculate regex for every symbol
    all_symbols_regex = []
    for i in range(len(all_symbols)):
        all_symbols_regex.append(determine_regex(all_symbols[i]))

    for i in range(len(all_symbols)):
        expressions = take_expressions_symbol(data_filename, all_symbols_regex[i], all_symbols_regex, exclusive)
        if(len(expressions) > 0):
            print("generating plot for:", all_symbols[i])
            output_filename = path_to_output_file + dataset + '-' + str(len(expressions)) + '-' + all_symbols[
                i] + '.svg'
            encodings = get_encodings(encoder, expressions)
            mean, stdev = calc_mean_stdev(encodings)
            print(str((stdev > 0.1).sum()))
            print(str(find_min_max(stdev, False, 4)))
            plot_mean_stdev(mean, stdev, output_filename)




# vorläufige main-funktion
if __name__ == "__main__":

    dataset = 'oneVarPoly13'
    path_to_trained_set = 'results_all/'
    path_to_data_file = 'semvec-data/expressions-synthetic/'
    path_to_output_file = 'results_all/' + dataset
    path_to_output_file_symbols = 'results_all/' + dataset + '/per_symbol'

    encoder_filename = path_to_trained_set + 'rnnsupervisedencoder-' + dataset + '.pkl'
    data_filename = path_to_data_file + dataset + '.json.gz'

    #make a new directory in results for the outputfiles
    try:
        os.mkdir(path_to_output_file)
    except OSError:
        print("Creation of the directory %s failed" % path_to_output_file)
    else:
        print("Successfully created the directory %s " % path_to_output_file)

    # make a new directory in results for the outputfiles
    try:
        os.mkdir(path_to_output_file_symbols)
    except OSError:
        print("Creation of the directory %s failed" % path_to_output_file_symbols)
    else:
        print("Successfully created the directory %s " % path_to_output_file_symbols)

    path_to_output_file = path_to_output_file + '/'
    path_to_output_file_symbols = path_to_output_file_symbols + '/'
    
    encoder = load_encoder(encoder_filename)

    plot_for_all_eqClasses_average(encoder, data_filename, dataset, path_to_output_file)
    plot_for_every_eqClass(encoder, data_filename, dataset, path_to_output_file)

    symbol_exclusively = False
    all_symbols = ["And", "Or", "Not", "*", "-", "+", "**"]
    plot_by_symbol(encoder, data_filename, dataset, path_to_output_file_symbols, all_symbols, symbol_exclusively)


