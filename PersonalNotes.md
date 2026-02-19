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

#### How does this effect the models
* High Kurtosis = many outliers => results in MSE loss exploding. Outliers dominate training
* increases Overfitting risk

## QQ plots

The Quantile-Quantile method(QQ) is plot technique to determine whether dataset is following a particular distribution(Usually normal) or not. Whether two samples of data came from the same population or not.

#### How to read a QQ plot

1. Points follow a straight line => data is approximately normal.

2. S-Shape Curve => Upward curve : Positive skew, downward curve: negative skew

3. Ends bend away from line => high Kurtosis/heavy tails.

4. Flat middle, curved ends => Non normal distribution.

#### Effects on model
* Neural networks and tree models do not require normality.
* It answers `Should I transform my target before modeling?`


























