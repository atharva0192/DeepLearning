---
Comments: Quick Recap of ML concepts
---
### Types
- #### Supervised Learning
	- Labeled training data
	- Goal to correctly label new data
	- Example
		- Image Classification
- #### Unsupervised Learning
	- Unlabeled Data
	- Goal to categorize the observations
	- Example
		- Clustering
- #### Reinforcement Learning
	- Unlabeled Data
	- System Receives feedback
	- Goal to perform better actions
- #### Self Supervised or Predictive Learning
	- Use part of data to predict other parts of data

#### Measure Success for Classification
- Precision
	- % of positive labels that are correct
	- TP / TP+FP
- Recall
	- % of positive labels that are correctly labeled
	- TP / TP+FN
- Accuracy
	- % of correct labels
	- TP+TP/samples


### Classification
- Statistical Learning Framework
	- y = f(x)
- K Nearest Neighbor
	- Non Parametric Method
- Linear Classifier 
	- y = wx + b
	- Bias Trick
- Support Vector Machines
	- Linear SVMs
		- Hyperplane that maximizes the margin between the positive and negative plane 
		- For Separable
			- Quadratic optimization problem
			- w = a . y . x where x are support vectors and a  are learned weights
			- Classification - sign(f(x)) = sign(a . y. x .x + b)
		- Non - Separable data:
			- Objective to minimize misclassifications as well
			
	- Non Linear SVMs
		- Map non separable data to higher dimension
		- Kernel Trick
			- Instead of explicitly computing the lifting function define a kernel function
				- K(x,y) = phi(x).phi(y) where phi is lifting function
			- Polynomial Kernel
			- Gaussian Kernel
			- Histogram Intersection
			- Square root
			
	- Multiclass SVMs Loss 
	
- Softmax Classifier

- Non Linear Classifier
	- Good accuracy on challenging models
	- Two ways to make Non Linear predictors from Linear Classifiers
		- Shallow Approach - Feature Transformation and then Linear Classifier
		- Deep Approach - Stack multiple layers of Linear Classifiers
- Perceptron
	- Non Linearity or Activation Functions
- Two Layer Neural Network
	- Bigger Hidden Layer the more Expensive the model
- Deep Pipeline
	- Each Layer extracts features from output of previous layer


- Bias Variance Tradeoff
	- Bias - Error due to simplifying model assumptions
	- Variance - Error due to randomness of training example
	- Simple Model - High Bias  , Low Variance - Underfitting - Training and Test error high
	- Complex Model - Low Bias , High Variance - Overfitting - Training error low but high test error


### Neural Network

#### Image 
- One way to  stretch pixels in single column vector
	- Problems : 
		- High Dimensionality
		- Local Relationship
	- Solution : CNN
- Layer of Convolution Layer
	- Input layer
	- Convolutional Layer
	- Non Linearity Layer (Sigmoid , Tanh ,Relu)
	- Pooling Layer (Max Pooling )
	- Fully Connected Layer
	- Classification Layer (Softmax)

#### Conv Layer
- Input:
	- W1 x D1 x H1
	- Requires four parameters
		- Number of filters K
		- their spacial extent F
		- the stride S (Jumps after filter)
		- amount of zero padding P (one in left and one in right)
	 
- Output
	-  W2 x H2 x D2
		- W2 = (W1 - F + 2P)/S+1
		- H2 = (H1  - F + 2P)/S+1
		- D2 = K


#### Pooling Layer
- Input:
	- W1 x D1 x H1
	- Requires four parameters
		- Their Spatial Extent 
		- the stride
		

- Output
	-  W2 x H2 x D2
		- W2 = (W1 - F )/S+1
		- H2 = (H1  - F )/S+1
		- D2 = K

