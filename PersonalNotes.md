# Concepts I have faced and Understanding them in depth

## Skewness vs Kurtosis


## Skewness
Skewness measures the asymmetry in the distribution. It tell you wheter data leans more to the left or right.

#### **Types of skewness**
-> Positive skewness(Right skew): More data is towards the left side. Long tail on the right.
	- Mean > median
-> Negative Skewness(Left skew): Long tail on left side, where as vast amount data leaning towards the right.
	- Mean < median.

#### How does this effect the models
- Many models assumes normality. Coefficient become unstable.
- prediction biased toward tail
- Loss dominated by extreme values.

NNs don't assume normality - But skewness still matters because
- Larger values dominates the gradient by targeting loss functions.
- focus more will be on rare extremes
- Tree models - least sensitive - split based on order, not distribution.

## Kurtosis
Kurtosis measure how heavy the tails and peakness of a distribution are. Length and height. 
- it tells you how extreme the outliers might be.

#### Types of Kurtosis
-> High kurtosis(Leptokurtic)
- Sharp peak | Heavy tails | More extreme outliers. 

-> Low Kurtosis (Platykurtic)
- Flatter shape | Fewer extreme values