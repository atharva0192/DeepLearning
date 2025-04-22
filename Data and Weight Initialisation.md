
## Dataset Preparation
- Train Test Validation
- K Fold Validation

## Data Preprocessing
- Zero Mean Data

## Weight Initialization
- Constant
	- All neurons will produce same output and undergo same changes
	- No Asymmetry
- Small Random with mean 0
	- Works well for small network
	- Diminshing Gradient Problem
- Small random with mean 0 and sd 1
	- Almost all either -1 or 1 
	- so Gradients will be zero
- Xavier
	-  np.random.randn(fanin , fanout) / sqrt(fan_in)
	- Breaks with RELU Nonlinearity
- Xavier improved
	- np.random.randn(fan_in,fan_out) / sqrt(2/fan_in)


## Monitor and Visualise Loss Curve
 ![[Pasted image 20250306012141.png]]
 ![[Pasted image 20250306012233.png]]![[Pasted image 20250306012257.png]]