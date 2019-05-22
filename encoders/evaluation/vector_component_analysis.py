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


def plot_per_symbol(output_filename, symbol, error):
    """

    output_filename -- the path for the plot
    """

    # plot for symbol at position i
    # switch to the given figure
    plt.figure(symbol)

    # set labels and save the figure
    plt.xlabel('component')
    plt.ylabel('value')

    #plt.ylim(ymax=0.04)

    x = np.arange(len(error))

    # plt.plot(x, all_error[i], linestyle='', linewidth=0.1, marker='o')
    plt.bar(x, error)

    # plt.errorbar(x, mean, yerr=stdev, linestyle='', marker='o')

    plt.savefig(output_filename)

    # close the figure
    plt.close(symbol)


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




def calculate_center_for_every_eqClass(encoder, data_filename):
    """
    Calculates the mean representation of every equivilance class

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    """

    eqClass_mean = list()

    eqClasses, expressionsByEqClass = take_expressions_eq_classes(data_filename)
    for i in range(len(eqClasses)):
        encodings = get_encodings(encoder, expressionsByEqClass[i])
        mean, stdev = calc_mean_stdev(encodings)
        eqClass_mean.append(mean)

    return eqClass_mean


def calculate_average_distance_to_center_for_eqClass_by_symbol(nameOfEqclass, average, encodedExpressions, all_symbols_regex, exclusive):
    """
    calculates the average distance to the mean representation of the eqClass

    nameOfEqclass -- name to compare to symbol
    average -- mean representation of eqClass
    encodedExpressions -- expressions of eqClass
    all_symbols_regex -- symbols for comparison
    return -- average distance to center of given eqClass
    """

    # nof expressions per symbol
    nofExpressions = np.zeros(len(all_symbols_regex))

    #error per symbol
    error = np.zeros((len(all_symbols_regex), encoder.get_representation_vector_size()))

    #calculate for every symbol if it is in equivilance class
    symbolInEqClass = list()
    for i in range(len(all_symbols_regex)):
        if (exclusive):
            inExpression = symbol_is_only_symbol(nameOfEqclass, all_symbols_regex[i], all_symbols_regex)
        else:
            inExpression = symbol_is_in_expression(nameOfEqclass, all_symbols_regex[i])
        if(inExpression):
            symbolInEqClass.append(True)
        else:
            symbolInEqClass.append(False)

    # iterate over all expressions in eqClass
    for i in range(len(encodedExpressions)):
        # iterate over all symbols
        for j in range(len(all_symbols_regex)):
            if(symbolInEqClass[j]):
                nofExpressions[j] += 1
                error[j] += np.absolute(average - encodedExpressions[i])

    #calculate average per symbol
    for i in  range(len(error)):
        if(nofExpressions[i] > 0):
            error[i] = error[i] / nofExpressions[i]

    return error


def calculate_average_distance_to_center_for_eqClass(average, encodedExpressions):
    """
    calculates the average distance to the mean representation of the eqClass

    average -- mean representation of eqClass
    encodedExpressions -- expressions of eqClass
    """

    error = np.zeros(encoder.get_representation_vector_size())

    # iterate over all expressions in eqClass
    for i in range(len(encodedExpressions)):
        error += np.absolute(average - encodedExpressions[i])

    error = error/len(encodedExpressions)

    return error



def plot_average_distance_per_symbol(encoder, data_filename, dataset, path_to_output_file, all_symbols, exclusive):
    """
    Plots a bar plot of the average distance of the elements of the equivilance classes
    to the center of the equivilance class

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    dataset --  name of the used dataser
    path_to_output_file -- path for the output file
    all_symbols -- all possible symbols in a dataset
    """

    # calculate regex for every symbol
    all_symbols_regex = []
    for i in range(len(all_symbols)):
        all_symbols_regex.append(determine_regex(all_symbols[i]))

    #initialize the overall error for every symbol and every slot
    #usually representation_vector_size is 64
    all_error = np.zeros((len(all_symbols_regex), encoder.get_representation_vector_size()))

    #calculate the center of every equivilance class
    print("calculating average")
    centers = calculate_center_for_every_eqClass(encoder, data_filename)

    #number of equivilance classes that contain the symbol
    nofClasses = np.zeros(len(all_symbols))
    nofExpressions = np.zeros(len(all_symbols))

    eqClasses, expressionsByEqClass = take_expressions_eq_classes(data_filename)
    #iterate over eqClasses
    for i in range(len(eqClasses)):
        print("calculating for eqClass", eqClasses[i])
        encodedExpressionsByEqClass = get_encodings(encoder, expressionsByEqClass[i])

        error = calculate_average_distance_to_center_for_eqClass_by_symbol(eqClasses[i], centers[i], encodedExpressionsByEqClass, all_symbols_regex, exclusive)

        #check if class contributed to symbol
        for j in range(len(error)):
            if (np.sum(error[j]) > 0):
                nofClasses[j] += 1
                nofExpressions[j] += len(expressionsByEqClass[i])
        all_error = np.add(all_error, error)

    #average over number of equivilace classes
    for i in range(len(all_error)):
        if(nofClasses[i] > 0):
            for j in  range(len(all_error[i])):
                all_error[i][j] = all_error[i][j] / nofClasses[i]

    #plotting
    for i in range(len(all_symbols_regex)):
        if(np.sum(all_error[i]) > 0):
            print("plotting for", all_symbols[i])
            filename = path_to_output_file + dataset + '-' + str(int(nofExpressions[i])) + '-' + str(int(nofClasses[i])) + '_' + all_symbols[i] + '.svg'
            symbol = all_symbols[i]
            error = all_error[i]
            plot_per_symbol(filename, symbol, error)


def plot_average_distance_all(encoder, data_filename, dataset, path_to_output_file):
    """
    Plots a bar plot of the average distance of the elements of the equivilance classes
    to the center of the equivilance class (average over all eqClasses)

    encoder -- the encoder object for encoding
    data_filename -- the path to the file with the data
    dataset --  name of the used dataser
    path_to_output_file -- path for the output file
    """

    #usually representation_vector_size is 64
    all_error = np.zeros(encoder.get_representation_vector_size())

    #calculate the center of every equivilance class
    print("calculating average")
    centers = calculate_center_for_every_eqClass(encoder, data_filename)

    eqClasses, expressionsByEqClass = take_expressions_eq_classes(data_filename)
    #iterate over eqClasses
    for i in range(len(eqClasses)):
        print("calculating for eqClass", eqClasses[i])
        encodedExpressionsByEqClass = get_encodings(encoder, expressionsByEqClass[i])
        error = calculate_average_distance_to_center_for_eqClass(centers[i], encodedExpressionsByEqClass)
        all_error = np.add(all_error, error)

    all_error = all_error / len(eqClasses)

    #plotting
    print("plotting for all")
    filename = path_to_output_file + dataset + '-all.svg'
    plot_per_symbol(filename, "all_expr", all_error)







# vorläufige main-funktion
if __name__ == "__main__":

    #poly5
    #oneVarPoly13
    dataset = 'oneVarPoly13'
    path_to_trained_set = 'results/'
    path_to_data_file = 'semvec-data/expressions-synthetic/'
    path_to_output_file = 'results/' + dataset
    path_to_output_file_symbols = 'results/' + dataset + '/symbol_Manfred'

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

    #plot_for_all_eqClasses_average(encoder, data_filename, dataset, path_to_output_file)
    #plot_for_every_eqClass(encoder, data_filename, dataset, path_to_output_file)

    symbol_exclusively = False
    all_symbols = ["And", "Or", "Not", "*", "-", "+", "**"]
    #plot_by_symbol(encoder, data_filename, dataset, path_to_output_file_symbols, all_symbols, symbol_exclusively)

    plot_average_distance_per_symbol(encoder, data_filename, dataset, path_to_output_file_symbols, all_symbols, symbol_exclusively)
    plot_average_distance_all(encoder, data_filename, dataset, path_to_output_file_symbols)
