#pragma once
#include <vector>

using namespace std;

namespace MachineLearning{

	class perceptron
	{
	public:
		perceptron(float eta,int epochs);
		float netInput(vector<float> X);
		int predict(vector<float> X);
		void fit(vector< vector<float> > X, vector<float> y);
		void printErrors();
		void exportWeights(string filename);
		void importWeights(string filename);
		void printWeights();
	private:
		float m_eta;
		int m_epochs;
		vector < float > m_w;
		vector < float > m_errors;
	};

	class AdalineGD
	{
	public:
		AdalineGD(float eta, int epochs);
		float netInput(vector<float> X);
		int predict(vector<float> X);
		void fit(vector< vector<float> > X, vector<float> y);
		void printErrors();
		void exportWeights(string filename);
		void importWeights(string filename);
		void printWeights();
	private:
		float m_eta;
		int m_epochs;
		vector < float > m_w;
		vector < float > m_errors;
	};

	class LogisticRegression
	{
	public:
		LogisticRegression(float eta, int epochs);
		float netInput(vector<float> X);
		int predict(vector<float> X);
		void fit(vector< vector<float> > X, vector<float> y);
		void printErrors();
		void exportWeights(string filename);
		void importWeights(string filename);
		void printWeights();
	private:
		float m_eta;
		int m_epochs;
		vector < float > m_w;
		vector < float > m_errors;
	};

	class NeuralNetworkMLP
	{
	public:
		NeuralNetworkMLP(vector<int> topology, float l1, float l2, int n_epochs, float eta, float alpha, float decrease_const, bool shuffle);
		void printWeights();
		void printOutput();
	private:
		//Structs and typedef
		struct Neuron 
		{
			Neuron(float val, int index);
			float m_val;
			int m_index;
		};
		typedef vector<Neuron> Layer;

		//Functions
		float sigmoid(float z);
		float sigmoid_gradient(float z);
		void feedForward(vector<float> X);
		float L1_reg(float lambda, vector< vector <float> > weights);
		float L2_reg(float lambda, vector< vector <float> > weights);
		float get_cost(vector<float> y, vector<float> output);
		int predict(vector<float> X);
		void get_gradient(vector<float> X, vector<float> y);

		//TODO: get_gradient(), fit()

		//Variables
		vector<Layer> network;
		vector< vector<float> > m_w;
		float l1;
		float l2;
		int n_epochs;
		float eta;
		float alpha;
		float decrease_const;
		bool shuffle;
		vector< vector<float> > m_Deltas;
	};

	//Utility functions

	void printMatrix(vector< vector<float> > a);
	void printVector(vector<float> a);

}