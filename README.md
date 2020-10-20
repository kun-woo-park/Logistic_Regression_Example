# Logistic_Regression_Example
This code is a simple Example for logistic regression using pytorch.
 
## Define problem
The problem is that i should diagnose either benign or malignant at breast cancer. i offered some data sets which measured at Wisconsin within 9 features(Clump Thickness, Uniformity of Cell Size, etc.)  to predict for it. 
 
## Confirm data
First, i should check about the given data, and confirm whether it has missing values or not. Then i typed some codes below to do for it.
```python
# Check raw data

df=pd.read_csv("Train_Data.txt",names=colnames)
print(df)
```
Then i found some missing values in data, and i replace them with their mean value for each column(feature).
```python
# Delete rows with nan values for restore

nan_value = float("NaN")
df.replace("?", nan_value, inplace=True)
non_missed_df=df.dropna(axis=0)
non_missed_df

# Check rows with nan values

is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
rows_with_NaN

# Restore missing values

mean_value_BN=int(round(np.mean(list(map(int,
                                         non_missed_df["Bare Nuclei"].values)))))
df.replace(nan_value, mean_value_BN, inplace=True)
```

## Model define
I defined logistic regression model like below.
```python
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x): 
        pred = torch.sigmoid(self.linear(x))
        return pred   #probability (not direct value)

    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0
```

## Train model
I trained model with BCE and using SGD optimizer. After train, finaly, i could get 100% accuracy at test data set.
```python
# instantitate optimizer 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, weight_decay=0.001) # Set L2 regularize term lambda as 0.1

# training the model 
epochs = 2000
losses = []

for epoch in range(epochs):
    y_pred = model.forward(x_train)

    # calculrate loss 
    loss = criterion(y_pred, y_train)
    if (epoch%100==0):
        print("epoch: ", epoch, "loss: ", loss.item())
    losses.append(loss.item())

    optimizer.zero_grad() # clear gradients wrt parameters
    loss.backward()
    optimizer.step()
```
