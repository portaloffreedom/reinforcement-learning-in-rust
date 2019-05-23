# Download data, unzip, etc.
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st




# Set some parameters to apply to all plots. These can be overridden
# in each plot if desired
import matplotlib
# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

_, ax = plt.subplots()

# Define a function for the line plot with intervals
def lineplotCI(x_data, y_data, low_CI, upper_CI, minimum, maximum,  x_label, y_label, title, color, file_name):
    # Create the plot object

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 3, color = color, alpha = 1, label = file_name)
    ax.plot(x_data, minimum, lw=1, color=color, alpha=1, label='5% quantile')
    ax.plot(x_data, maximum, lw=1, color=color, alpha=1, label='95% quantile')
    # Shade the confidence interval
    ax.fill_between(x_data, low_CI, upper_CI, color=color, alpha=0.1, label='25-75 quantile')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')

def add_plot(csv_name, color):
    dataset = pd.read_csv(csv_name, header=None)
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    upper = mean + std
    lower = mean - std

    upper_quantile = dataset.quantile(0.75)
    median = dataset.quantile(0.5)
    lower_quantile = dataset.quantile(0.25)

    max_quantile = dataset.quantile(0.95)
    min_quantile = dataset.quantile(0.05)

    lower_interval, upper_interval = st.t.interval(0.95, 99, loc=mean, scale=std)
    # Call the function to create plot
    # lineplotCI(x_data = list(range(0, 400))
    #            , y_data = median
    #            , low_CI=lower_quantile
    #            , upper_CI=upper_quantile
    #            , minimum = min_quantile
    #            , maximum = max_quantile
    #            , x_label='Episodes'
    #            , y_label='Value of Policy'
    #            , title='Value of policy over time'
    #            , color=color)

    lineplotCI(x_data=list(range(0, 400))
               , y_data=mean
               , low_CI=lower
               , upper_CI=upper
               , minimum=min_quantile
               , maximum=max_quantile
               , x_label='Episodes'
               , y_label='Value of Policy'
               , title='Value of policy over time'
               , file_name=csv_name
               , color=color)



# add_plot("q_learning_epsilon_rewards.csv", '#539caf')
add_plot("q_learning_epsilon_rewards.csv", '#999111')
add_plot("double_q_epsilon_rewards.csv", '#990a11')
plt.show()