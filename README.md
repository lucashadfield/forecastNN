# forecastNN

Tools to easily create and train a seq2seq model on a pandas DataFrame of data.


### Initiate and Fit
```python
from forecastNN.seq2seq import  Seq2SeqForecaster

model = Seq2SeqForecaster(pred_steps=10)
model.compile()
model.fit(train_df, epochs=20)

```

### Predict
```python
model.predict(test_df)
```

### Plot Examples
```python
model.plot_example_fit(test_df, loc=0) #Seq2SeqForecaster
```

![Prediction Example](https://i.imgur.com/65q9OCi.jpg "Example from model.plot_example_fit(test_df)")


```python
model.plot_example_fit(test_df, loc=0, nstd=2) #Seq2SeqMonteCarloForecaster
```

![Prediction Example](https://i.imgur.com/Hy1XY5d.jpg "Example from model.plot_example_fit(test_df)")