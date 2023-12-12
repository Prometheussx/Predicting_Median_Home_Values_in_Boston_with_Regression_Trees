## Real Estate Price Prediction with Decision Tree Regression

### Overview

This Python code utilizes Decision Tree Regression for predicting real estate prices based on features such as bedrooms, square footage, and location. The model is trained and evaluated using a dataset imported from a CSV file.

### Libraries

- pandas
- numpy
- scikit-learn (DecisionTreeRegressor, train_test_split, metrics)
- matplotlib

### Dataset Import and Pre-Processing

1. The dataset ('real_estate_data.csv') is loaded into a pandas DataFrame.
2. Data exploration and cleanup, including handling missing values.

### Pre-Processing

1. Dropping rows with missing values.
2. Separating features (X) and target variable (Y).

### Train-Test Split

1. Splitting the dataset into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
```

### Regression Tree

1. Creating a Decision Tree Regression model.
2. Training the model on the training set.
3. Making predictions on the test set.

```python
regression_tree = DecisionTreeRegressor(criterion="squared_error")
regression_tree.fit(X_train, Y_train)
prediction = regression_tree.predict(X_test)
```

### Evaluation

1. Calculating the Mean Absolute Error (MAE) for prediction accuracy.
2. Printing the model's score on the test set.

```python
mae = (prediction - Y_test).abs().mean() * 1000
print(f"Mean Absolute Error: ${mae:.2f}")
print("Model Score:", regression_tree.score(X_test, Y_test))
```
## License

This project is released under the [MIT License](https://github.com/Prometheussx/Kaggle-Notebook-Cancer-Prediction-ACC96.5-With-Logistic-Regression/blob/main/LICENSE).


## Author

- Email: [Email_Address](mailto:erdemtahasokullu@gmail.com)
- LinkedIn Profile: [LinkedIn Profile](https://www.linkedin.com/in/erdem-taha-sokullu/)
- GitHub Profile: [GitHub Profile](https://github.com/Prometheussx)
- Kaggle Profile: [@erdemtaha](https://www.kaggle.com/erdemtaha)

Feel free to reach out if you have any questions or need further information about the project.


### Note

Ensure the 'real_estate_data.csv' file is in the correct path or update the file path accordingly.

