import gc # Garbage collector
import os  # For handling directory paths
import glob # To match the patern for file paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle # For saving lists
from scipy import stats
import itertools # For creting combinations of things

# # for google colab
# from google.colab import drive
# drive.mount("/content/drive")

# # Specify the directory containing the CSV files
# directory = "/content/drive/MyDrive/Mining/harth"

directory = "/home/ceidass/Documents/Projects/Mining/Project/harth"

# Create directories if they do not exist
values_dir = directory.replace("/harth", "") + "/values"
intervals_dir = directory.replace("/harth", "") + "/intervals"
overlabels_dir = directory.replace("/harth", "") + "/overlabels"
nn = directory.replace("harth","nn")

os.makedirs(values_dir, exist_ok=True)
os.makedirs(intervals_dir, exist_ok=True)
os.makedirs(overlabels_dir, exist_ok=True)
os.makedirs(nn, exist_ok=True)
# Names of the six features of each dataset
feature_names = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
required_columns = feature_names + ["label"]

# Get all the CSV file paths
csv_files = glob.glob(os.path.join(directory, '*.csv'))
nn_f = glob.glob(os.path.join(nn, '*.npy'))

# Functions for manually calculating the correlation matrix with less data plus their sequentials steps
def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

def weighted_covariance(x, y, weights):
    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(y, weights)
    return np.sum(weights * (x - mean_x) * (y - mean_y)) / np.sum(weights)

def weighted_correlation(x, y, weights):
    cov_xy = weighted_covariance(x, y, weights)
    std_x = np.sqrt(weighted_covariance(x, x, weights))
    std_y = np.sqrt(weighted_covariance(y, y, weights))
    return cov_xy / (std_x * std_y)




# for idx, file in enumerate(csv_files):
    
#     # Get the base name of the file and remove the '.csv' extension
#     base_name = os.path.basename(file).replace('.csv', '')
#     df = pd.read_csv(file)
    
#     df = df[required_columns] # Get only the columns we need

#     s = df.to_numpy()

#     # Isolate the last column to leave unchanged
#     unchanged_column = s[:, [-1]]  # Select the last column

#     # Remove the last column from the data to scale
#     data_to_scale = s[:, :-1]  # Select all columns except the last one

#     # Initialize the MinMaxScaler
#     scaler = MinMaxScaler()

#     # Fit the scaler to the data and transform it
#     scaled_data = scaler.fit_transform(data_to_scale)

#     # Concatenate the unchanged last column back to the scaled data
#     final_data = np.hstack((scaled_data, unchanged_column))
#     print(final_data)
#     np.save(nn+"/"+base_name+".npy", s)



    # Get the base name of the file and remove the '.csv' extension
    #base_name = os.path.basename(file).replace('.csv', '')

    # values = np.empty((0,len(feature_names)))
    # intervals = np.empty(0)
    # overlabels = np.empty(0)
    # sum = np.zeros(len(feature_names))

    # counter = 0 #initialize counter to measure the first occurence of a label
    # tempLabel = df["label"][0] # Templabel is the first label of the Dataframe
    # overlabels = np.append(overlabels,np.asarray([tempLabel]), axis=0) # Append the first value of the first label

    # for i in df.index:
    #     if df["label"][i] == tempLabel:
    #       sum += np.absolute(df.iloc[i, :-1].to_numpy())  # Add the new values to the sum (exclude last column which is label value)
    #       counter += 1  # Increase the counter
    #     else:
    #       intervals = np.append(intervals, np.asarray([counter]), axis=0)
    #       sum /= counter #Divide the sum with the number of added values
    #       values = np.append(values, np.expand_dims(sum, axis=0), axis=0)
    #       tempLabel = df["label"][i] # Re initialize temporal label
    #       overlabels = np.append(overlabels,np.asarray([tempLabel]), axis=0) # Append the last value of the last label
    #       sum = np.absolute(df.iloc[i,:-1].to_numpy()) #Fill the sum array with the dataframe row of sensor values (absolute values)
    #       counter = 1 #First occurence of the current label

    # Save the values in numpy form so we can load them when needed
    # We saved the transpose of the values array so in every vector we have all the values of the specific sensor's axis of the person 
    # np.save(directory.replace("/harth","")+"/values/"+base_name+"_val.npy",np.transpose(values))
    # np.save(directory.replace("/harth","")+"/intervals/"+base_name+"_int.npy",intervals)
    # np.save(directory.replace("/harth","")+"/overlabels/"+base_name+"_olab.npy",overlabels)

##################################################################################################################################

val_files = glob.glob(os.path.join(values_dir, '*.npy'))
inter_files = glob.glob(os.path.join(intervals_dir, '*.npy'))

valArr = np.empty((len(val_files),len(feature_names),0)) #3d array for storing all values

# We will now recreate the dataset, but for every sequenial activity we are BINNING (smoothing) the data with the absolute average


# for idx, vfile in enumerate(val_files): # Itterrate through files of value arrays and interval arrays 
#     # print(idx) # debug
#     arr = np.load(vfile) # Loading the saved value arrays
#     print(arr.shape)#debug
#     ifile = directory.replace("harth","intervals/")+os.path.basename(vfile).replace('_val', '_int') # Loading the saved intevals arrays
#     inter = np.load(ifile)
#     #print(inter.shape)
#     length = np.sum(inter,dtype=np.int32) # Find the actual length of the timeseries by adding the intervals
#     #print(os.path.basename(vfile)) #debug
#     #print(length)#debug
#     helpArr = np.zeros((len(feature_names),length)) # Create a 2D array for storing a person's measurements (with the right length)

#     for i,axis in enumerate(feature_names): #Iterate through each sensor's values inside the value array 
#         count = 0 #Counter of the cell in helpArr[i]
#         measure = 0 # Counter of the cell in arr[i] (specific measure)
#         while measure < arr.shape[1] and count < length - inter[measure]:

#             # print(count) #debug
#             # print(measure) #debug
#             for j in range(int(inter[measure])):
#                 helpArr[i,count+j] = arr[i,measure]
            
#             measure += 1
#             count += j

#         if helpArr.shape[1] > valArr.shape[2]:
#             #print(helpArr.shape)#debug
#             tempArr = np.zeros((len(val_files),len(feature_names),helpArr.shape[1]))
#             tempArr[:,:,:valArr.shape[2]] = valArr[:,:,:] # replace the corresponding values
#             valArr = tempArr # Give to valArr the new shape
#             #print(valArr.shape)#debug

#         elif valArr.shape[2] > helpArr.shape[1]:
#             #print(valArr.shape) #debug
#             tempArr = np.zeros((len(feature_names),valArr.shape[2])) # Temporary array with the new dimensions
#             tempArr[:,:helpArr.shape[1]] = helpArr[:,:] # replace the corresponding values
#             helpArr = tempArr
#             #print(helpArr.shape)#debug
        
#         valArr[idx,i,:] = helpArr[i,:] #Pass the vector in the values array

# np.save(directory+"/newset.npy",valArr) # Save the new binned dataset
        
# # Delete the array
# del valArr

# # Run the garbage collector
# gc.collect()


#####################################################################################################################################

# data = np.load(directory+"/newset.npy") # Loading the smoothed dataset array
# new_data = [[None] * data.shape[1] for _ in range(data.shape[0])] #Empty list with size of the number of persons * features
# inter_data = [None] * len(inter_files)

# # Plot the 6 features for the selected person

# os.makedirs(directory.replace("harth","firstplots/"), exist_ok=True)

# # Arrays for storing the actual means and standard deviations
# means = np.empty((len(val_files),len(feature_names)))
# std = np.empty((len(val_files),len(feature_names)))

# os.makedirs(directory.replace("harth","metrics"), exist_ok=True)


# for idx ,values in enumerate(val_files):

#     base_name = os.path.basename(values).replace('.npy', '')
#     plt.figure(num=idx,figsize=(7, 8))

#     for i,fname in enumerate(feature_names):

#         #Clip zeros from the end if exists so they don't bother us in the plots
#         last_non_zero_index = -1
#         for j in range(len(data[idx,i]) - 1, -1, -1):
#             if data[idx,i,j] != 0:
#                 last_non_zero_index = j
#                 break

#         # Normalize data
#         scaler = MinMaxScaler() 
#         new_data[idx][i] = scaler.fit_transform(data[idx,i,:last_non_zero_index].reshape(-1,1))
        
#         row_mean = np.mean(new_data[idx][i]) #Mean Value of the feature
#         row_std = np.std(new_data[idx][i])  #Standard deviation of the feature

#         # Store the above values to their arrays
#         # Those metrics are not the actual metrics of the dataset but what have been created after the smoothing
#         means[idx,i] = row_mean
#         std[idx,i] = row_std

#         positions = [1,3,5,2,4,6]
#         plt.subplot(3, 2, positions[i])
#         plt.plot(new_data[idx][i], label=fname)
#         plt.axhline(y=np.mean(new_data[idx][i]), color='red', linestyle='--', label='Mean')

#         # Plot the variance area
#         plt.fill_between(range(len(new_data[idx][i])), 
#                          row_mean - row_std, 
#                          row_mean + row_std, 
#                          color='red', 
#                          alpha=0.2, 
#                          label='\u00B1 St.D.')

#         plt.title(fname)
#         plt.xlabel('Time Steps')
#         plt.ylabel('Normalized Value')
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig(directory.replace("harth","firstplots/")+base_name+".png")
#     plt.close() # Close the current figure after saving it, to free memory space

# # Save the means and StD arrays
# np.save(directory.replace("harth","metrics")+"/means.npy",means)
# np.save(directory.replace("harth","metrics")+"/std.npy",std)

# # Save list of arrays with the restored and smoothed data
# with open('restored_vals.pkl', 'wb') as f:
#     pickle.dump(new_data, f)


# # Delete array to free memory space
# del data
# del means
# del std
# gc.collect()

#######################################################################################################################################

#                    ################ CORRELATION MATRIX AND HEATMAP (means)##################################


# # Restore the saved list of arrays
# with open('restored_vals.pkl', 'rb') as f:
#     new_data = pickle.load(f)


# for i,file in enumerate(val_files):
#     new_data[i] = np.load(file)
#     int_name = os.path.basename(file).replace('val.npy', 'int.npy')
#     inter_data[i] = np.load(intervals_dir+"/"+int_name)


# # Number of features (measurements)
# n_features = new_data[0].shape[0]

# # Initialize a matrix to store pairwise correlations
# correlation_matrix = np.zeros((n_features, n_features))

# for idx in range(len(new_data)):
#     # Compute pairwise correlations for features 
#     for i in range(n_features):
#         for j in range(i, n_features):
#             if i == j:
#                 correlation_matrix[i, j] += 1.0 * np.sum(inter_data[idx])

#             else:
#                 # Extract those 2 feature values
#                 feature_i_values = new_data[idx][i, :]
#                 feature_j_values = new_data[idx][j, :]
#                 # Compute weighted correlation
#                 corr = weighted_correlation(feature_i_values, feature_j_values, inter_data[idx])
#                 correlation_matrix[i, j] += corr * np.sum(inter_data[idx])
#                 correlation_matrix[j, i] += corr * np.sum(inter_data[idx])

#     print(np.sum(inter_data[idx]))

# # Sum of all intervals of all persons
# intersum = 0
# for i in range(len(inter_data)):
#     intersum += np.sum(inter_data[i])


# correlation_matrix /= intersum
# print(correlation_matrix)#debug
# # Create the heatmap of the correlation matrix
# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(correlation_matrix, xticklabels=feature_names, yticklabels=feature_names, cmap='Reds', annot=True, fmt=".2f")

# # Set labels
# ax.set_xlabel('X Axis Labels')
# ax.set_ylabel('Y Axis Labels')

# # Save the heatmap as a PNG file
# plt.savefig(directory.replace("harth","firstplots")+"/avgcorrmat.png")
# plt.close()


#################################################################################################################################


# means = np.load(directory.replace("harth","metrics")+"/means.npy")
# std = np.load(directory.replace("harth","metrics")+"/std.npy")

# # Transpose so the means of same features are in rows
# temp = np.transpose(means)


#            ################################ T TEST (means)##################################


# ttmatrix = np.empty((2,means.shape[1],means.shape[1]))

# for i in range(means.shape[1]):
#     for j in range(i, means.shape[1]):

#         # Make the 2 arrays have the same variance (This test assumes that the populations have identical variances by default.)
#         var1 = np.var(temp[i])
#         var2 = np.var(temp[j])

#         # Determine the scaling factor (sf)
#         sf = np.sqrt(var1 / var2)  # sqrt(variance_ratio)
#         mean_j = np.mean(temp[j])
#         scaled_tempj = (temp[j] - mean_j) * sf + mean_j

#         ttmatrix[0,i,j], ttmatrix[1,i,j] = stats.ttest_ind(temp[i],scaled_tempj)
#         ttmatrix[0,j,i] = ttmatrix[0,i,j]
#         ttmatrix[1,j,i] = ttmatrix[1,i,j]
    
# # # print(ttmatrix[0]) #debug
# # # print(ttmatrix[1]) #debug


# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

# # Plot the heatmaps
# # Plot the heatmaps using seaborn
# sns.heatmap(np.absolute(ttmatrix[0]), ax=ax1, xticklabels=feature_names, yticklabels=feature_names, cmap="Reds", annot=ttmatrix[0], fmt=".2f")
# sns.heatmap(ttmatrix[1] * 100, ax=ax2, xticklabels=feature_names, yticklabels=feature_names, cmap="Reds", annot=True, fmt=".2f")

# # Rotate the vertical axis tick labels to be horizontal
# ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
# ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

# # Adjust the space between the subplots
# plt.subplots_adjust(hspace=0.5)

# # Add titles
# ax1.set_title("T Statistic Matrix")
# ax2.set_title("P-values matrix (x100)")


# plt.savefig(directory.replace("harth","firstplots")+"/ttest_hm.png")
# plt.close()

################ ANOVA TEST (Analysis of Variances) (means) ##################################

# # Calculate the means and variances of each population (row)
# means = np.mean(temp, axis=1)
# variances = np.var(temp, axis=1)

# # Choose the reference variance (e.g., the average variance)
# reference_variance = np.mean(variances)

# # Calculate scaling factors to adjust variances to the reference variance
# scaling_factors = np.sqrt(reference_variance / variances)

# # Adjusting the sets to have the same variance and preserving their means
# temp = np.array([(temp[i] - means[i]) * scaling_factors[i] + means[i] for i in range(temp.shape[0])])

# #List made from the arrays of means
# meansLists = [temp[i] for i in range(temp.shape[0])]

# # List of items
# items = list(range(6))

# # Generate combinations
# for r in range(2, 7):
#     comb = list(itertools.combinations(items, r))
#     for indices in comb:
#         selected = [meansLists[i] for i in indices]
#         f_val, p_val = stats.f_oneway(*selected)
#         print(indices)
#         if p_val>0.75  and (f_val<2 and f_val>-2):
#             print("#######################################################################################################")
#         print(f_val) #debug
#         print(p_val) #debug
#         print(" ")


#################################################################################################################################

## Restore the remaked dataset
# newset = np.load(directory+"/newset.npy")

# #Matrix for storing the average T Test matrix for Time series
# tttsmatrix = np.zeros((2,len(feature_names), len(feature_names)))
# sumsums = 0

# for idx in range(newset.shape[0]):

#     for i in range(newset[idx].shape[0]):

#         #Clip zeros from the end if exists so they don't bother us in the plots
#         last_non_zero_index = -1
#         for l in range(newset.shape[2] - 1, -1, -1): # Start from the end and decreasing until reaching the 0
#             if newset[idx,i,l] != 0:
#                 last_non_zero_index = l
#                 break

#         for j in range(i, newset[idx].shape[0]):
#             vari = np.var(newset[idx,i,:last_non_zero_index])
#             varj = np.var(newset[idx,j,:last_non_zero_index])

#             # Determine the scaling factor (sf)
#             sf = np.sqrt(vari / varj)  # sqrt(variance_ratio)
#             mean_j = np.mean(newset[idx,j,:last_non_zero_index])
#             scaled_dataj = (newset[idx,j,:last_non_zero_index] - mean_j) * sf + mean_j

#             t_stat , p_value = stats.ttest_ind(newset[idx,i,:last_non_zero_index],scaled_dataj)
#             # print(t_stat)
#             # print(p_value)
#             # print("")
#             tttsmatrix[0,i,j] += t_stat * newset[idx,i,:last_non_zero_index].shape[0]
#             tttsmatrix[1,j,i] += p_value * newset[idx,i,:last_non_zero_index].shape[0]
#             if i!=j:
#                 tttsmatrix[0,j,i] += tttsmatrix[0,i,j]
#                 tttsmatrix[1,j,i] += tttsmatrix[1,i,j]
#     sumsums += newset[idx,i,:last_non_zero_index].shape[0]
#     print(sumsums)#debug

# tttsmatrix /= sumsums

# print(tttsmatrix[0])
# print(tttsmatrix[1])



# # Find the mean of the intervals
# interlist = np.empty((len(values_dir),0))
# intersum = 0
# intercount = 0
# for i,inter in enumerate(inter_files):
#     file = np.load(inter)
#     intercount += file.shape[0]
#     for j in range(file.shape[0]):
#         print(file[j])
#         intersum += file[j]
    
# intermean = intersum / intercount
# print(intermean) # ABOUT 920


window = 100
step = 50
dt = np.empty((0,100,6))
lb = np.empty((0,12))
counter = 0
tl = np.zeros((1,12))

for idx, f in enumerate(nn_f):

    print(idx)#debug
    print(f)
    file = np.load(f)
    for i in range(0,file.shape[0]-1-window , step):

        temp = file[i:i+window,:6]
        exp = temp.reshape((1,)+temp.shape)

        # Add new sample
        dt = np.append(dt,exp, axis=0)
        
        # Create one hot encoding
        l = file[i+window,6]
        match l:
            case 1:
                tl[0,0] = 1
            case 2:
                tl[0,1] = 1
            case 3:
                tl[0,2] = 1
            case 4:
                tl[0,3] = 1
            case 5:
                tl[0,4] = 1
            case 6:
                tl[0,5] = 1
            case 7:
                tl[0,6] = 1
            case 8:
                tl[0,7] = 1
            case 13:
                tl[0,8] = 1
            case 14:
                tl[0,9] = 1
            case 130:
                tl[0,10] = 1
            case 140:
                tl[0,11] = 1
            
            
        lb = np.append(lb,tl,axis=0)
        tl = np.zeros((1,12))
        counter += 1 

np.save(nn+"/data.npy",dt)
np.save(nn+"/labels.npy",lb)
print(counter)#debug

del dt
del lb
gc.collect()