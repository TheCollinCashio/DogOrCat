##  Dog or Cat?
As a starter to work my way into the "AI" world, I started off with a simple example of a convolutional neural network that decides whether an input image was a cat or a dog.

#   The Models
I was interested in seeing which method outperforms the other. For the first model, I used one convolutional layers with one pooling layer. For the second model, I used three convolutional layers with three pooling layers. For the fourth model, I used four convolutional layers with four pooling layers.

**Model 1:**
![Model 1](https://github.com/TheCollinCashio/DogOrCat/blob/master/Metrics/Model1.png)
**Model 2:**
![Model 2](https://github.com/TheCollinCashio/DogOrCat/blob/master/Metrics/Model2.png)
**Model 3:**
![Model 3](https://github.com/TheCollinCashio/DogOrCat/blob/master/Metrics/Model3.png)

#   The Training Data
| Set of Images | # of Images   |
| ------------- |--------------:|
| Test Set      | 350           |
| Training Set  | 800           |

#   The Metrics
| Study AVG Epo | Model 1       | Model 2       | Model 3       |
| ------------- |:-------------:|:-------------:| -------------:|
| Accuracy      | 99.61%        | 99.66%        | 99.38%        |
| Loss          | 1.314%        | 1.034%        | 1.862%        |
| Val Accuracy  | 99.96%        | 99.35%        | 99.51%        |
| Val Loss      | 0.223%        | 2.345%        | 1.268%        |

#   Analysis
According to the metrics given above, Model 2 had better accuracy. Although, Model 1 has the best accuracy for the test data set given.
