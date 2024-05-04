# Save this as test_sklearn.py
from sklearn.linear_model import Ridge
import scipy

# Create a Ridge regression model instance
reg = Ridge(alpha=.1)
print("Ridge model:", reg)
