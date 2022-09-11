# What is PCA 

![Olivetti dataset](./figures/olivetti_dataset.png)
![Imperfect reconstruction](./figures/imperfect_reconstruction.png)
![Out of dataset reconstruction](./figures/ood_reconstruction.png)
![Top six principle components and directions](./figures/principle_components.png)
![Out of dataset explanation](./figures/span_explanation.jpg)
![4 datapoint example](./figures/4_datapoints.png)
![5000 datapoint example](./figures/5000_datapoints.png)
![PCA basis](./figures/pca_basis.png)
![3d example](./figures/3d.png)
![Degenerate data](./figures/degenerate.png)
![PCA dropoff](./figures/pca_dropoff.png)



Principle Component Analysis (PCA) is a fundamental method used to reduce the dimensionality of a problem. This is desirable for many reasons most notably improving computational tracability. 

Once in the lower dimensional space machine learning techniques can be used as normal such as clustering, neural networks etc. The value of the principle components denotes how important/much information is given by one of the orthogonal directions in this new basis. Therefore, we reduce dimensions by discarding directions with small principle components/importantance. 

I hope that this article is different to other PCA articles on medium as it motivates the problem algebraicly aswell as geometrically which is not something I have seen elsewhere.

#  Representing data

Multivariate data can be represented by a matrix, $\mathbf{A}$, with each row being a datapoint and each column being a different parameter. The covariance matrix of the data is proportional to $\mathbf{A
^TA}$. This fact is true as can be seen below for the case where the mean is zero for all params

$$\text{data} = \mathbf{A^TA} = \begin{bmatrix} x_1 & x_2 & x_3 \\ y_1 & y_2 & y_3 \end{bmatrix}\begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ x_3 & y_3 \end{bmatrix} = \begin{bmatrix} \sum_i x_i^2 & \sum_i x_iy_i \\ \sum_i y_ix_i & \sum_i y_i^2 \end{bmatrix} $$


# Covariance in PCA


The singular value decomposition (SVD) is a generalisation of the eigendecomposition for non-square matrices and for an $M \times N$ matrix is given by $\mathbf{U \Sigma V^T}$ with $\mathbf U$ having shape $M \times M$, $\mathbf \Sigma$ having shape $M \times N$ and $\mathbf V$ having shape $N \times N$. Using the SVD for the data and the fact that the covariance is proportional to $\mathbf{A^T A}$, we show the following:

$$\text{data} = \mathbf{A} = \mathbf{U \Sigma V^T}$$
$$\text{covar(data)} \propto \mathbf{A^TA} = \mathbf{V \Sigma^T U^T U \Sigma V^T} = \mathbf{V \Sigma^T \Sigma V^T}$$
$$\text{Since } \mathbf U \text{ is unitary so } \mathbf{U^T U} = \mathbf I$$



Comparing to equation x we see that this is the eigendecomposition for the covariance. The $\Sigma$ matrix is non-square but $\Sigma^T \Sigma$ is square as shown. 

$$ \mathbf{\Sigma^T \Sigma} = \begin{bmatrix} \sigma_1  & 0 & 0 \\ 0 & \sigma_2 & 0 \end{bmatrix} \begin{bmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}$$

The eigenvalues of the covariance matrix are the singular values squared, therefore there is an equivilance between the singular values and standard deviation. 


So the principle components of the data are the standard deviations (i.e. the square root of the variance) in the orthogonal directions given by the eigenvectors of the covariance matrix (why this?).  

stress here that the singular value is therefore the same as the standard deviation and that is important! why does this mean more information in this direction? why is stdev information.

explain that SVD is actually quicker in reality to fin

- V is the matrix of eigenvectors/principle component directions
- want square root of the eigenvalues of the covariance matrix
- want to eigenvectors of the covariance matrix 
- happens to be given by the SVD and is much quicker to run so from now on do it this way 

# In 2D 

- four points
- change in the y = x and y = -x directions 
- y = x twice as big 
- see that the eigenvectors are as expected and then the singular values too
- go to many points that have the same underlying correlation but on greater scale (how to call these Gaussians?)

In a toy example in 2D there are four points plotted which can be represented by the data matrix below. Plotting the eigenvectors of the covariance matrix scaled by the standard deviation (singular value) in that direction we get the following basis vectors.


$$\text{data} = \mathbf A = \begin{bmatrix} 1 & 1 \\ -1 & -1 \\ 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}$$

It is possible to transform the coordinates of the data matrix into the principle component basis 

V A V^T

We see that if we do the same but for a few hundred correlated gaussian data points the same principle components and directions are seen.

# Reducing the dimensions

- give an example of points that are on a straight line
- then say same thing in 3d
- project onto the largest two principle components

- do the 2d plot 
- do the projection into 2d plot

As mentioned earlier, PCA is such a powerful tool as you just get rid of the principle component directions that do not carry that much information (small standard deviation/singular value). 

In two dimensions, if all the points lie on a line, there is a set relationship between the x and y variables, meaning one of them is redundant. In the case below where y = x, we can reduce to a single dimension by projecting the points into the principle component frame y=x and discard the second as the singular value is zero in the orthogonal y=-x direction.

Looking at some data in three dimensions and plotting the scaled principle component directions, we see that there is one direction that doesn't carry nearly as much information as the other two. In this case you just project the datapoints onto the principle component directions of the larger two singular values only.


# eigenfaces

The eigenface dataset has 400 images that are taken from 40 people and is a great way to visualise PCA in higher dimensions. Each image is grayscale 64 x 64, so can be flattened into a 4096 dimension vector with values ranging from 0 to 1 for each pixel, below is an example image. As done previously, the principle components and their directions can be found from the SVD of the data matrix. 


You can project each image onto the principle component directions to get the image's coordinates in principle component space

$$\text{image} = \sum_i \sigma_i \mathbf v_i $$


These directions in 4096 dimensional space can be turned back into images, the examples with the six largest principle values can be seen below. 


Again, to transform into this basis you project the image vector onto each of the principle component directions. Equally, you can reconstruct the image vector by projecting then image vector in principle component basis onto the pixel space basis vectors (i.e. each pixel is a direction). 

But as discussed earlier, principle components with small values offer little information, so we can discard some. Below is how the face is reproduced with different number of principle components used. For example, if we use just 100 principle components we compress the image ~41 times whilst still having a recognisable face. (not counting the face that we have the store the principle components and their directions).

# Reconstruction out of the dataset

For the eigenfaces dataset there are only 400 principle components and directions as the rank of the matrix is 400. 

This means there are 400 basis vectors to describe a 4096 dimensional space, so the this basis does not span the entire space of 64 x 64 images. Though we fully span the space of images in the dataset as this is how we came up with the principal component directions in the first place. 

For this reason, it is not possible to completely reproduce an image outside of the dataset even when using all principle components. 

Below are two examples of how well out principle components reproduce it. Although neither are perfect, the out of dataset face is reproduced much more faithfully than the image of the car. The sketch below shows how the space spanned by the principle components has much more over lap with the out of dataset face than the car, perhaps as you may expect. 

![span_explanation.jpg](attachment:https://github.com/hws1302/pca-vs-autoencoder/span_explanation.jpg)
