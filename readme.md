# Stock price prediction from news headline using RNN
> [https://github.com/ranakroychowdhury/Multimodal-Deep-Learning-based-Stock-Price-Prediction.git](https://github.com/ranakroychowdhury/Multimodal-Deep-Learning-based-Stock-Price-Prediction.git)

Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices, company performance data and news data suitable for machine learning algorithms to process.
Can we actually predict stock prices with machine learning? Investors make educated guesses by analyzing data. They read the news, study the company history, industry trends, market behavior and other data that go into making a prediction. The prevailing theories render stock prices to be random and unpredictable.
This project utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict stock prices. For data with timeframes, recurrent neural networks (RNNs) come in handy but current research have shown that LSTM networks are the most popular and useful variants of RNNs.
We have used Keras to build a LSTM to predict stock prices using historical stock prices and news data and visualize both the predicted price values over time and the optimal parameters for the model.

## Data Descriptio
The dataset consists of:
Top 5 world news headlines from **Kaggle** And The opening, closing, adjusted closing, high and low price for 15 companies from **Yahoo Finance** for approximately eight years (from 08-08-2008 to 01-07-2016).The 15 American based multinational technology companies used in our data are as follows:

	1.Baidu Inc
	2.Adobe Systems Incorporated
	3.Oracle Corporation
	4.Amazon.com, Inc.
	5.Alphabet Inc Class C
	6.NVIDIA Corporation
	7.Microsoft Corporation
	8.NetEase Inc (ADR)
	9.Electronic Arts Inc.
	10.Apple Inc.
	11.QUALCOMM, Inc.
	12.Cisco Systems, Inc.
	13.Texas Instruments Incorporated
	14.Intel Corporation
	15.IBM Common Stock

The implementation details and the result of the project can be found in Report.docx and Results.docx respectively.

## How to contribute

1. `./data.py`: �������� tokenlize, �ɼ����� Normalize
2. `./train.py`: ѵ��ģ�ͣ��õ� `model.h5`
3. `./test.py`: ����ģ��
4. `./tune.py`: ���ò�ͬģ�ͳ�����������ѵ��ģ�ͣ��Ա�Ч��