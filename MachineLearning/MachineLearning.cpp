#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <math.h> 
#include <stdlib.h>
#include <time.h>

#include "MachineLearning.h"

using namespace std;

namespace MachineLearning{

	////////// PERCEPTRON ////////////////////////////////

	perceptron::perceptron(float eta, int epochs)
	{
		m_epochs = epochs;
		m_eta = eta;
	}

	void perceptron::fit(vector< vector<float> > X, vector<float> y)
	{
		for (int i = 0; i < X[0].size() + 1; i++) // X[0].size() + 1 -> I am using +1 to add the bias term
		{
			m_w.push_back(0);
		}
		for (int i = 0; i < m_epochs; i++)
		{
			int errors = 0;
			for (int j = 0; j < X.size(); j++)
			{
				float update = m_eta * (y[j] - predict(X[j]));
				for (int w = 1; w < m_w.size(); w++){ m_w[w] += update * X[j][w - 1]; }
				m_w[0] = update;
				errors += update != 0 ? 1 : 0;
			}
			m_errors.push_back(errors);
		}
	}

	float perceptron::netInput(vector<float> X)
	{
		// Sum(Vector of weights * Input vector) + bias
		float probabilities = m_w[0];
		for (int i = 0; i < X.size(); i++)
		{
			probabilities += X[i] * m_w[i + 1];
		}
		return probabilities;
	}

	int perceptron::predict(vector<float> X)
	{
		return netInput(X) > 0 ? 1 : -1; //Step Function
	}

	void perceptron::printErrors()
	{
		printVector(m_errors);
	}

	void perceptron::exportWeights(string filename)
	{
		ofstream outFile;
		outFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			outFile << m_w[i] << endl;
		}

		outFile.close();
	}

	void perceptron::importWeights(string filename)
	{
		ifstream inFile;
		inFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			inFile >> m_w[i];
		}
	}

	void perceptron::printWeights()
	{
		cout << "weights: ";
		for (int i = 0; i < m_w.size(); i++)
		{
			cout << m_w[i] << " ";
		}
		cout << endl;
	}

	////////// ADALINE GD ////////////////////////////////

	AdalineGD::AdalineGD(float eta, int epochs)
	{
		m_epochs = epochs;
		m_eta = eta;
	}

	void AdalineGD::fit(vector< vector<float> > X, vector<float> y)
	{
		for (int i = 0; i < X[0].size() + 1; i++) // X[0].size() + 1 -> I am using +1 to add the bias term
		{
			m_w.push_back(0);
		}
		for (int i = 0; i < m_epochs; i++)
		{
			int errors = 0;
			for (int j = 0; j < X.size(); j++)
			{
				float update = m_eta * (y[j] - netInput(X[j]));
				for (int w = 1; w < m_w.size(); w++){ m_w[w] += update * X[j][w - 1]; }
				m_w[0] = update;
				errors += (y[j] - predict(X[j])) != 0 ? 1 : 0;
			}
			m_errors.push_back(errors);
		}
	}

	float AdalineGD::netInput(vector<float> X)
	{
		// Sum(Vector of weights * Input vector) + bias
		float probabilities = m_w[0];
		for (int i = 0; i < X.size(); i++)
		{
			probabilities += X[i] * m_w[i + 1];
		}
		return probabilities;
	}

	int AdalineGD::predict(vector<float> X)
	{
		return netInput(X) > 0 ? 1 : -1; //Step Function
	}

	void AdalineGD::exportWeights(string filename)
	{
		ofstream outFile;
		outFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			outFile << m_w[i] << endl;
		}

		outFile.close();
	}

	void AdalineGD::importWeights(string filename)
	{
		ifstream inFile;
		inFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			inFile >> m_w[i];
		}
	}

	void AdalineGD::printWeights()
	{
		cout << "weights: ";
		for (int i = 0; i < m_w.size(); i++)
		{
			cout << m_w[i] << " ";
		}
		cout << endl;
	}

	void AdalineGD::printErrors()
	{
		printVector(m_errors);
	}

	////////// LOGISTIC REGRESSION //////////////////////////////

	LogisticRegression::LogisticRegression(float eta, int epochs)
	{
		m_epochs = epochs;
		m_eta = eta;
	}

	void LogisticRegression::fit(vector< vector<float> > X, vector<float> y)
	{
		for (int i = 0; i < X[0].size() + 1; i++) // X[0].size() + 1 -> I am using +1 to add the bias term
		{
			m_w.push_back(0);
		}
		for (int i = 0; i < m_epochs; i++)
		{
			int errors = 0;
			for (int j = 0; j < X.size(); j++)
			{
				float update = m_eta * (y[j] - netInput(X[j]));
				for (int w = 1; w < m_w.size(); w++) { m_w[w] += update * X[j][w - 1]; }
				m_w[0] = update;
				errors += (y[j] - predict(X[j])) != 0 ? 1 : 0;
			}
			m_errors.push_back(errors);
		}
	}

	float LogisticRegression::netInput(vector<float> X)
	{
		// Sum(Vector of weights * Input vector) + bias
		float probabilities = m_w[0];
		for (int i = 0; i < X.size(); i++)
		{
			probabilities += X[i] * m_w[i + 1];
		}
		return probabilities / (1 + abs(probabilities));
	}

	int LogisticRegression::predict(vector<float> X)
	{
		return netInput(X) > 0 ? 1 : -1; //Step Function
	}

	void LogisticRegression::exportWeights(string filename)
	{
		ofstream outFile;
		outFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			outFile << m_w[i] << endl;
		}

		outFile.close();
	}

	void LogisticRegression::importWeights(string filename)
	{
		ifstream inFile;
		inFile.open(filename);

		for (int i = 0; i < m_w.size(); i++)
		{
			inFile >> m_w[i];
		}
	}

	void LogisticRegression::printWeights()
	{
		cout << "weights: ";
		for (int i = 0; i < m_w.size(); i++)
		{
			cout << m_w[i] << " ";
		}
		cout << endl;
	}

	void LogisticRegression::printErrors()
	{
		printVector(m_errors);
	}

	/////////// NEURAL NETWORK MULTI LAYER PERCEPTRON ///////////

	NeuralNetworkMLP::Neuron::Neuron(float val, int index)
	{
		m_val = val;
		m_index = index;
	}

	NeuralNetworkMLP::NeuralNetworkMLP(vector<int> topology, float l1, float l2, int n_epochs, float eta, float alpha, float decrease_const, bool shuffle)
	{
		//Default Values vector<int> topology, float l1=0.0, float l2=0.0, int n_epochs=500, float eta=0.001, float alpha=0.0, float decrease_const=0.0, bool shuffle=true
		l1 = l1;
		l2 = l2;
		n_epochs = n_epochs;
		eta = eta;
		alpha = alpha;
		decrease_const = decrease_const;
		shuffle = shuffle;
		srand(unsigned(time(NULL)));

		//Initialize Hidden Layer Vector //vector<int> hidden_layers
		for (int i = 0; i < topology.size(); i++)
		{
			network.push_back(Layer());
			for (int j = 0; j < topology[i]; j++)
			{
				network[i].push_back(Neuron(0.0, i == 0 || i == topology.size() ? 0 : 1));
			}
		}

		// Add biases
		for (int l = 1; l < network.size(); l++)
		{
			network[l].insert(network[l].begin(), Neuron(1.0, 0));
		}

		//Initialize Random Weights
		for (int i = 0; i < network.size() - 1; i++)
		{
			m_w.push_back(vector<float>{});
			for (int j = 0; j < network[i].size() * (network[i+1].size()-1); j++)
			{
				m_w[i].push_back(((float)rand() / (RAND_MAX)) - 1 - ((int)rand() % 2 + (-1))); // random demical numbers from -1 to 1
			}
		}

	}

	float NeuralNetworkMLP::sigmoid(float z)
	{
		return 1 / (1 + exp(-z));
	}

	float NeuralNetworkMLP::sigmoid_gradient(float z)
	{
		return sigmoid(z) * (1 - sigmoid(z));
	}
	
	void NeuralNetworkMLP::feedForward(vector<float> X)
	{
		for (int l = 0; l < network.size(); l++) // Loop through each layer
		{
			for (int n = l == 0 ? 0 : 1; n < network[l].size(); n++) // Loop through each Neuron
			{
				if (l == 0)
				{
					network[l][n].m_val = X[n];
				}
				else
				{
					float value = 0.0;
					for (int i = 0; i < network[l - 1].size(); i++)
					{
						value += network[l - 1][i].m_val * m_w[l - 1][i*n];
					}
					network[l][n].m_val = value;
				}
			}
		}
	}

	float NeuralNetworkMLP::L1_reg(float lambda, vector< vector <float> > weights)
	{
		float weightSum = 0.0;
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].size(); j++)
			{
				weightSum += abs(weights[i][j]); ///// Note: if err check [:, 1:] 
			}
		}
		return (lambda / 2.0) + weightSum;
	}

	float NeuralNetworkMLP::L2_reg(float lambda, vector< vector <float> > weights)
	{
		float weightSum = 0.0;
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].size(); j++)
			{
				weightSum += pow(weights[i][j], 2); ///// Note: if err check [:, 1:] 
			}
		}
		return (lambda / 2.0) + weightSum;
	}

	float NeuralNetworkMLP::get_cost(vector<float> y, vector<float> output)
	{
		float cost = 0.0;
		float term1 = 0.0;
		float term2 = 0.0;
		for (int i = 0; i < y.size(); i++)
		{
			term1 += -y[i] * log(output[i]); // Note: Remove ( - ) if there are errors
			term2 += (1 - y[i]) * log(1 - output[i]);
		}
		cost = (term1 - term2);
		float L1_term = L1_reg(l1, m_w);
		float L2_term = L1_reg(l2, m_w);
		cost = cost + L1_term + L2_term;
		return cost;
	}

	void NeuralNetworkMLP::get_gradient(vector<float> X, vector<float> y)
	{
		
	}

	int NeuralNetworkMLP::predict(vector<float> X)
	{
		feedForward(X);
		int prediction = 0;
		float minChance = 0.0;
		for (int i = 0; i < network[network.size() - 1].size(); i++)
		{
			if (network[network.size() - 1][i].m_val > minChance)
			{
				minChance = network[network.size() - 1][i].m_val;
				prediction = i;
			}
		}
		return prediction;
	}

	void NeuralNetworkMLP::printWeights()
	{
		printMatrix(m_w);
	}

	void NeuralNetworkMLP::printOutput()
	{
		cout << "Output: ";
		for (int i = 0; i < network[network.size() - 1].size(); i++)
		{
			cout << network[network.size() - 1][i].m_val << " ";
		}
		cout << endl;
	}

	////////// UTILITY FUNCTIONS ////////////////////////////////

	void printMatrix(vector< vector<float> > a)
	{
		for (int i = 0; i < a.size(); i++)
		{
			for (int j = 0; j < a[i].size(); j++)
			{
				cout << a[i][j] << " ";
			}
			cout << endl;
		}
	}

	void printVector(vector<float> a)
	{
		for (int i = 0; i < a.size(); i++)
		{
			cout << a[i] << endl;
		}
	}

}