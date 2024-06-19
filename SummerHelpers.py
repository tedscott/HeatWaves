from datetime import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a function that takes in the data for a specific lat/lon and returns the start/end,length of summer
# and generates a plot if requested
def SummerStats(temp_data, threshold, save_plot):
    '''

    Function that uses the input 75th percentile temperature in K to identify the summer from a time series of temperatures

    Args:
        temp_data: xarray of temperature data for a single grid cell
        threshold: xarray for the 75th percentile threshold to use
        save_plot: save out a png of the time series with threshold, summer start/end shown
        
    Returns:
        start day, end day, summer length
        
    Example:
        start_day, last_day, summer_length = SummerStats(input_temps, input_thresh, True)

    '''
    # extract DOY and T2m values for those days
    x1 = temp_data.time.dt.dayofyear.values
    y1 = temp_data.values

    #print(x1, y1)
    # polynomial fit, degree 3 as in Wang et al 2021
    coefs = np.polyfit(x1, y1, 3)
    poly = np.poly1d(coefs)

    ## *** TO-DO *** x_values needs to deal with leap years if not already handled
    x_values = np.linspace(min(x1), max(x1), 366) # create some x values for plotting polynomial fit

    # use fit to get stats
    summer_length = np.count_nonzero(poly(x_values) >= threshold.values)
    summer_days = np.where(poly(x_values) >= threshold.values)[0]

    # deal with grids that didn't have days above threshold
    if summer_length == 0:
        first_day = 0
        last_day = 0
    else: 
        first_day = x_values[min(summer_days)]
        last_day = x_values[max(summer_days)]

    # make and save plot if desired
    ##
    ##
    if save_plot:
        cell_lat = temp_data.lat.values
        cell_lon = temp_data.lon.values
        yr = temp_data.time.dt.year.values[0]

        plt.figure(figsize=(16,6))
        plt.plot(x1,y1, label='daily mean T2m')
        plt.plot(x_values, poly(x_values), label='fitted degree 3 polynomial', color='red')
        plt.axhline(threshold.values, color="black", linewidth=0.8, linestyle="dashed") 
        plt.axvline(first_day, color="black", linewidth=0.8, linestyle="dashed")
        plt.axvline(last_day, color="black", linewidth=0.8, linestyle="dashed")
        plt.annotate('summer = '+str(summer_length)+' days', xy=(first_day+5,np.mean(poly(x_values))), fontsize=18)
        plt.annotate('75th percentile = '+str(np.round(threshold.values,1))+'K', xy=(0,threshold.values+1), fontsize=18)
        plt.xlabel('DOY (' + str(yr) + ')')
        plt.ylabel('Mean Daily Temp (K)')
        plt.title("Summer for Lat,Lon = [" +str(cell_lat)+","+str(cell_lon)+ "] Baseline Years = (check data)")
        plt.legend()
        plt.savefig("polyfit_stats_"+str(cell_lat)+"_"+str(cell_lon)+"_"+str(yr)+".png")
        plt.show()

    # return
    return first_day, last_day, summer_length

# a function that takes in a dataset of global t2m values and if it is a leap year averages
# the temps on Feb 28 & Feb 29, then drops Feb 29 to ensure the year has 365 days
def HandleLeapYears(input_ds):
    '''

    Function that takes the mean of Feb 28 & Feb 29 (if it exists) and returns an xarray Dataset that has 365 days
    where Feb 28 will now have a t2m value that is the mean of Feb 28 & Feb 29

    Args:
        input_ds: the data set containing a single year of data that may need adjusting for leap year
        
    Returns:
        output_ds
        
    Example:
        no_leap_ds = HandleLeapYears(input_ds)

    '''
    

    # check if a leap year
    if(input_ds.time.dt.is_leap_year[0].values):
        
        # make deep copy
        output_ds = input_ds.copy(deep=True)
        
        # get mean of Feb 28 (time index 58) & Feb 29 (time index 59) for each grid cell and overwrite Feb 28 with it
        mean_dat = output_ds.t2m[58:60].mean(dim='time', skipna=True, keep_attrs=True)
        output_ds.t2m[58] = mean_dat
        
        # drop Feb 29
        output_ds = output_ds.convert_calendar('noleap')

        # return the new dataset 
        return output_ds
    else:
        return input_ds
    