from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

# Loading the functional datasets

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      data.func[0])  # 4D data

##############################################################################
# Extract time series
from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=0)

#time_series = masker.fit_transform(data.func[0],
#                                   confounds=data.confounds)
time_series = masker.fit_transform(fmri_filenames, confounds=[x])[:,0:90]
##############################################################################
# Compute the sparse inverse covariance
from sklearn.covariance import GraphLassoCV
estimator = GraphLassoCV()

estimator.fit(time_series)

##############################################################################
# Display the connectome matrix
from matplotlib import pyplot as plt

# Display the covariance
plt.figure(figsize=(10, 10))

# The covariance can be found at estimator.covariance_
plt.imshow(estimator.covariance_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
# And display the labels
x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
y_ticks = plt.yticks(range(len(labels)), labels)
plt.title('Covariance')

##############################################################################
# And now display the corresponding graph
from nilearn import plotting
coords = atlas.region_coords

plotting.plot_connectome(estimator.covariance_, coords,
                         title='Covariance')


##############################################################################
# Display the sparse inverse covariance (we negate it to get partial
# correlations)
plt.figure(figsize=(10, 10))
plt.imshow(-estimator.precision_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
# And display the labels
x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
y_ticks = plt.yticks(range(len(labels)), labels)
plt.title('Sparse inverse covariance')

##############################################################################
# And now display the corresponding graph
plotting.plot_connectome(-estimator.precision_, coords,
                         title='Sparse inverse covariance')

plotting.show()
