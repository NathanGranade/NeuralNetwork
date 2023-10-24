package neuralNetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.apache.commons.lang3.ArrayUtils;

public class NeuralNetworkTest 
{
	// Initialize variables for node sizes, learning rate, weights, biases, hidden, and output layer
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double ETA;

    private double[][] weightsHidden;
    private double[][] weightsOutput;
    
    private double[] biasHidden;
    private double[] biasOutput;

    private double[] hiddenLayer;
    private double[] outputLayer;
	int correct = 0; int total = 0; int correct0 = 0; int total0 = 0; int correct1 = 0; int total1 = 0; int correct2 = 0; int total2 = 0; int correct3 = 0; int total3 = 0; int correct4 = 0; int total4 = 0; int correct5 = 0; int total5 = 0; int correct6 = 0; int total6 = 0; int correct7 = 0; int total7 = 0; int correct8 = 0; int total8 = 0; int correct9 = 0; int total9 = 0;


    public NeuralNetworkTest(int inputSize, int hiddenSize, int outputSize, double ETA) 
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.ETA = ETA;
        
        this.weightsHidden = new double[inputSize][hiddenSize];
        this.weightsOutput = new double[hiddenSize][outputSize];
        
        this.biasHidden = new double[hiddenSize];
        this.biasOutput = new double[outputSize];

        this.hiddenLayer = new double[hiddenSize];
        this.outputLayer = new double[outputSize];
        
        // Initialize weights and biases with random values
        Random rand = new Random();
        
        // Initialize weights
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHidden[i][j] = rand.nextDouble(1 - (-1)) - 1;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsOutput[i][j] = rand.nextDouble(1 - (-1)) - 1;
            }
        }

        // Initialize biases
        for (int i = 0; i < hiddenSize; i++) {
            biasHidden[i] = rand.nextDouble(1 - (-1)) - 1;
        }

        for (int i = 0; i < outputSize; i++) {
            biasOutput[i] = rand.nextDouble(1 - (-1)) - 1;
        }
    }

    // Function for evaluating the sigmoid
    public double sigmoid(double x) 
    {
        return 1 / (1 + Math.exp(-x));
    }
    
    // Function for evaluating the sigmoid derivative
    public double sigmoidDerivative(double x) 
    {
        return x * (1 - x);
    }
    
    // Function for shuffling the data within the mini-batches
    private void shuffleData(double[][] input, double[][] output) 
    {
        int n = input.length;
        int m = output.length;
        Random rand = new Random();
        
        for (int i = n - 1; i > 0; i--) 
        {
            int j = rand.nextInt(i + 1);
            // Swap input
            double[] tempInput = input[i];
            input[i] = input[j];
            input[j] = tempInput;
        }
        for (int i = m - 1; i > 0; i--)
        {
        	int j = rand.nextInt(i + 1);
            // Swap output
            double[] tempOutput = output[i];
            output[i] = output[j];
            output[j] = tempOutput;
        }
    }
    
    // Function for feeding forward through the network
    public void feedForward(double[] inputData) 
    {
    	// Dot product for hidden layer
        for (int i = 0; i < hiddenSize; i++) 
        {
            hiddenLayer[i] = 0;
            for (int j = 0; j < inputSize; j++) 
            {
            	// Multiply each input by each weight at hidden layer
                hiddenLayer[i] += inputData[j] * weightsHidden[j][i];
            }
            // Apply sigmoid function to each hidden layer + the bias at that layer
            hiddenLayer[i] = sigmoid(hiddenLayer[i] + biasHidden[i]);
        }
        
    	// Dot product for output layer
        for (int i = 0; i < outputSize; i++) 
        {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenSize; j++) 
            {
            	// Multiply each hidden layer input by each weight at output layer
                outputLayer[i] += hiddenLayer[j] * weightsOutput[j][i];
            }
            // Apply sigmoid function to each output layer + the bias at that layer
            outputLayer[i] = sigmoid(outputLayer[i] + biasOutput[i]);
        }
    }
    
    public void countGuesses(double[] output)
    {

		// Get the highest activation function (closest to one) aka our max
		// Compare position of max to position of "1" in target output (compare guess to correct digit)
		// Store correct and total guesses for printing later
		double max = outputLayer[0];
		
		for(int k = 0; k < outputLayer.length; k++) 
		{         	
			if(max < outputLayer[k])
			{
				max = outputLayer[k];
			}       
		}
		
		int correctGuess = ArrayUtils.indexOf(output, 1);
		int networkGuess = ArrayUtils.indexOf(outputLayer, max);
		
		if(correctGuess == 0) { total0++; if(networkGuess == 0) { correct0++; correct++; }}
		else if(correctGuess == 1) { total1++; if(networkGuess == 1) { correct1++; correct++; }}
		else if(correctGuess == 2) { total2++; if(networkGuess == 2) { correct2++; correct++; }}
		else if(correctGuess == 3) { total3++; if(networkGuess == 3) { correct3++; correct++; }}
		else if(correctGuess == 4) { total4++; if(networkGuess == 4) { correct4++; correct++; }}
		else if(correctGuess == 5) { total5++; if(networkGuess == 5) { correct5++; correct++; }}
		else if(correctGuess == 6) { total6++; if(networkGuess == 6) { correct6++; correct++; }}
		else if(correctGuess == 7) { total7++; if(networkGuess == 7) { correct7++; correct++; }}
		else if(correctGuess == 8) { total8++; if(networkGuess == 8) { correct8++; correct++; }}
		else if(correctGuess == 9) { total9++; if(networkGuess == 9) { correct9++; correct++; }}
		total++;
    }

    public void train(double[][] input, double[][] output, int epochs, int batchSize) 
    {
        int dataLength = input.length;
        
        // Loop for training the data for x amount of epochs
        for (int epoch = 0; epoch < epochs; epoch++) 
        {
        	// Initialize all variables for total/correct guesses to 0
        	correct = 0; total = 0; correct0 = 0; total0 = 0; correct1 = 0; total1 = 0; correct2 = 0; total2 = 0; correct3 = 0; total3 = 0; correct4 = 0; total4 = 0; correct5 = 0; total5 = 0; correct6 = 0; total6 = 0; correct7 = 0; total7 = 0; correct8 = 0; total8 = 0; correct9 = 0; total9 = 0;
       
 	
        	/////////////////////////////////
            // Stochastic Gradient Descent // 
            /////////////////////////////////
        	
        	// 1a. Randomize the order of the items in the training set.
        	shuffleData(input, output);
        	
        	// 2a. Divide the training set into equal sized mini-batches
            for (int start = 0; start < dataLength; start += batchSize) 
            {
                int end = start + batchSize;
                
                // Initialize variables for holding the gradients for each set of weights and biases at the end of each epoch
                double[][] gradientWeightsOutput = new double[outputSize][hiddenSize]; double[][] gradientWeightsHidden = new double[hiddenSize][inputSize];
                double[] gradientBiasHidden = new double[hiddenSize]; double[] gradientBiasOutput = new double[outputSize];
                
                // Loop for each input row based on the mini-batch
                for (int i = start; i < end; i++) 
                {
                	// 3a. Using back propagation, compute the weight gradients and bias gradients over the current mini-batch
                	
                	/////////////////////
                    // Backpropagation // 
                    /////////////////////
                	
                	// 1b. Using the current weights and biases [which are initially random] along with an 
                	// input vector X, compute the activations (outputs) of all neurons at all layers of the network. 
                	// This is the “feed forward” pass.
                    feedForward(input[i]);
                    
                    countGuesses(output[i]);

                    // Initialize variables for holding the gradients for weights/biases at the end of each mini batch
                	// And variables for holding error at hidden/output layers
                    gradientWeightsOutput = new double[outputSize][hiddenSize]; gradientWeightsHidden = new double[hiddenSize][inputSize];
                    gradientBiasHidden = new double[hiddenSize]; gradientBiasOutput = new double[outputSize];
                    double[] outputError = new double[outputSize]; double[] hiddenError = new double[hiddenSize];
                    
                    // 2b. Using the computed output of the final layer together with the desired output vector Y,
                    // Compute the gradient of the error at the final level of the network and then move “backwards”
                    // through the network computing the error at each level, one level at a time. This is the
                    // “backwards pass”.
                    for (int j = 0; j < outputSize; j++) 
                    {
                        outputError[j] = (outputLayer[j] - output[i][j]) * sigmoidDerivative(outputLayer[j]);
                    }

                    for (int j = 0; j < hiddenSize; j++) 
                    {
                        for (int k = 0; k < outputSize; k++) 
                        {
                            hiddenError[j] += outputError[k] * weightsOutput[j][k];
                        }
                        hiddenError[j] *= sigmoidDerivative(hiddenLayer[j]);
                    }

                    // 3b. Return as output the gradient values for each weight and bias in the network.
                    for (int j = 0; j < outputSize; j++) 
                    {
                    	gradientBiasOutput[j] += outputError[j];
                        for (int k = 0; k < hiddenSize; k++) 
                        {
                        	gradientWeightsOutput[j][k] += hiddenLayer[k] * gradientBiasOutput[j];
                        }    
                    }

                    for (int j = 0; j < hiddenSize; j++) 
                    {
                    	gradientBiasHidden[j] += hiddenError[j];
                        for (int k = 0; k < inputSize; k++) 
                        {
                        	gradientWeightsHidden[j][k] += gradientBiasHidden[j] * input[i][k];
                        }                        
                    }    
            	}
                
                // 4a. After completing the mini-batch update the weights and biases
                for (int j = 0; j < outputSize; j++) 
                {
                    for (int k = 0; k < hiddenSize; k++) 
                    {
                    	weightsOutput[k][j] = weightsOutput[k][j] - (ETA / 2) * gradientWeightsOutput[j][k];
                    }
                    biasOutput[j] = biasOutput[j] - (ETA / 2) * gradientBiasOutput[j];
                }
                
                for (int j = 0; j < hiddenSize; j++) 
                {
                    for (int k = 0; k < inputSize; k++) 
                    {
                    	weightsHidden[k][j] = weightsHidden[k][j] - (ETA / 2) * gradientWeightsHidden[j][k];
                    }
                    biasHidden[j] = biasHidden[j] - (ETA / 2) * gradientBiasHidden[j];
                }   
                      
                // 5a. If additional mini-batches remain, return to Step 3
            }    
            
            // Print the statistics for the current epoch  
            DecimalFormat percent = new DecimalFormat();
            double correctd = correct; double totald = total;
            System.out.println("Epoch " + (epoch+1) + "/" + epochs + ":\n");
            System.out.println("0 = " + correct0 + "/" + total0 + "\t" + "1 = " + correct1 + "/" + total1 + "\t" + "2 = " + correct2 + "/" + total2 + "\t" + "3 = " + correct3 + "/" + total3 + "\t" + "4 = " + correct4 + "/" + total4 + "\t");
            System.out.println("5 = " + correct5 + "/" + total5 + "\t" + "6 = " + correct6 + "/" + total6 + "\t" + "7 = " + correct7 + "/" + total7 + "\t" + "8 = " + correct8 + "/" + total8 + "\t" + "9 = " + correct9 + "/" + total9 + "\t");
            System.out.println("Accuracy = " + correct + "/" + total + " = " + percent.format(correctd/totald*100) + "%\n");
            
            // 6a. If our stopping criteria have not been met, return to Step 1
        }
    }
    
    public static List<String[]> importData(String file) throws IOException 
    {	
    	List<String[]> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] row = line.split(",");
                data.add(row);
            }
        }
        return data;
    }
    
    public static void fillData(List<String[]> CSVData, double[][] input, double[][] output, int inputSize, int outputSize)
    {
    	// Fill training and testing input/output arrays
        int i = 0;
        for (String[] row : CSVData)
        {
        	double[] tempOutput = new double[outputSize];
        	double[] tempInput = new double[inputSize];
        	
        	// Extract first number as output, put a "1" in the corresponding index.
        	tempOutput[Integer.parseInt(row[0])] = 1;
        	output[i] = tempOutput;
        	
        	for (int j = 1; j < row.length - 1; j++)
        	{        		
        		// Rescales grayscale values from 0-255 to 0-1.
        		tempInput[j-1] = Double.parseDouble(row[j])/255;
        	}
        	input[i] = tempInput;
        	i++;
        }
    }
    
    public static StringBuilder toASCII(double[] input)
    {
		StringBuilder art = new StringBuilder();
		for (int j = 0; j < input.length; j++)
		{	            				
			if (input[j] == 0)
			{
				art.append(" ");
			}
			else
			{
				art.append("X");
			}
			if (j % Math.sqrt(input.length) == 0)
			{
				art.append("\n");
			}
		}
		
    	return art;
    }
    
    public static void main(String[] args) throws NumberFormatException, IOException 
    {
    	//////////////////////////////////////////////////////////////////////////////////////////////
    	/*
    	This program implements a 3-layer fully connected neural network
    	using the process of feed forward, stochastic gradient descent, and back propagation.
    	After being trained on the MNIST training dataset, this program should be able to recognize 
    	hand written digits 0-9 using the MNIST testing dataset with above 90% accuracy.
    	*/
    	//////////////////////////////////////////////////////////////////////////////////////////////
    	 
    	// Initializes number of nodes on each layer, learning rate, batch size, and epochs
        int inputSize = 784;
        int hiddenSize = 30;
        int outputSize = 10;
        double ETA = 0.5;
        int batchSize = 10;
        int epochs = 30;
        
        NeuralNetworkTest neuralNetwork = new NeuralNetworkTest(inputSize, hiddenSize, outputSize, ETA);
        
        // Import and format data
        // Create lists and arrays for training and testing data inputs and outputs
        List<String[]> trainingDataCSV = importData("mnist_train.csv");
        double[][] trainingInput = new double[trainingDataCSV.size()][inputSize];
        double[][] trainingOutput = new double[trainingDataCSV.size()][outputSize];
        
        List<String[]> testingDataCSV = importData("mnist_test.csv");
        double[][] testingInput = new double[testingDataCSV.size()][1];
        double[][] testingOutput = new double[testingDataCSV.size()][1];
        
        fillData(trainingDataCSV, trainingInput, trainingOutput, inputSize, outputSize);
        fillData(testingDataCSV, testingInput, testingOutput, inputSize, outputSize);
        
        ///////////////////////////////
        // Main menu for the program //
        ///////////////////////////////
        
        // Print first set of menu options
        System.out.println("[1] Train the network\n");
        System.out.println("[2] Load a pre-trained network\n");
        System.out.println("[0] Exit the program\n");
        
        // Get and store user input
        Scanner userInput = new Scanner(System.in);
        System.out.println("> ");
        int choice = userInput.nextInt();
        
        // If user input = 1, run first menu option (train network)
        if (choice == 1)
        {
        	neuralNetwork.train(trainingInput, trainingOutput, epochs, batchSize);
            
            System.out.println("Training Complete.\n");
        }
        // If user input = 2, run second menu option (load pre-trained network)
        
        else if(choice == 2)
        {
        	// Load weights file from predetermined location
        	String weightsHiddenFile = "weights.txt";
        	BufferedReader reader = new BufferedReader(new FileReader(weightsHiddenFile));
        	String line = "";
        	int row = 0;
        	// Read each line into a double[][] to hold the hidden layer weights
        	while((line = reader.readLine()) != null && row < neuralNetwork.weightsHidden.length)
        	{
        		String[] cols = line.split(",");
        		int col = 0;
        		for(String  c : cols)
        		{
        			neuralNetwork.weightsHidden[row][col] = Double.parseDouble(c);
        			col++;
        		}
        		row++;
        	}
        
        	row = 0;
        	// Read each line into a double[][] to hold the output layer weights
        	while((line = reader.readLine()) != null)
        	{
        		String[] cols = line.split(",");
        		int col = 0;
        		for(String  c : cols)
        		{
        			neuralNetwork.weightsOutput[row][col] = Double.parseDouble(c);
        			col++;
        		}
        		row++;
        	}
        	// Close the reader object
        	reader.close();
        }
        
        // If user input = 0, exit program.
        else if(choice == 0)
        {
        	System.out.println("Exiting Program...\n");
        	userInput.close();
            System.exit(0);  
        }
        
        // If user input not 0, 1, or 2, exit program with error msg.
        else
        {
        	System.out.println("Invalid choice.\n");
        	System.exit(0);
        }
        
        // Create loop for main menu
        while (true)
        {
        	// Print second layer menu options
        	System.out.println("[1] Train the network\n");
            System.out.println("[2] Load a pre-trained network\n");
            System.out.println("[3] Display network accuracy on TRAINING data\n");
            System.out.println("[4] Display network accuracy on TESTING data\n");
            System.out.println("[5] Run network on TESTING data showing images and labels\n");
            System.out.println("[6] Display the misclassified TESTING images\n");
            System.out.println("[7] Save the network state to file\n");
            System.out.println("[0] Exit the program\n");
            
            // Get and store user input
            System.out.println("> ");
            choice = userInput.nextInt();
            
            // Given user input, run appropriate menu option
            switch(choice)
            {
            
            // If user input = 1, run first menu option (train network)
            case 1:
            	neuralNetwork.train(trainingInput, trainingOutput, epochs, batchSize);
                
                System.out.println("Training Complete.\n");
                break;
                
            // If user input = 2, run second menu option (load pre-trained network)    
            case 2:            
	        	// Load weights file from predetermined location
	        	String weightsHiddenFile = "weights.txt";
	        	BufferedReader reader = new BufferedReader(new FileReader(weightsHiddenFile));
	        	String line = "";
	        	int row = 0;
	        	// Read each line into a double[][] to hold the hidden layer weights
	        	while((line = reader.readLine()) != null && row < neuralNetwork.weightsHidden.length)
	        	{
	        		String[] cols = line.split(",");
	        		int col = 0;
	        		for(String  c : cols)
	        		{
	        			neuralNetwork.weightsHidden[row][col] = Double.parseDouble(c);
	        			col++;
	        		}
	        		row++;
	        	}
	        
	        	row = 0;
	        	// Read each line into a double[][] to hold the output layer weights
	        	while((line = reader.readLine()) != null)
	        	{
	        		String[] cols = line.split(",");
	        		int col = 0;
	        		for(String  c : cols)
	        		{
	        			neuralNetwork.weightsOutput[row][col] = Double.parseDouble(c);
	        			col++;
	        		}
	        		row++;
	        	}
	        	// Close the reader object
	        	reader.close();
            
                break;
            
            // If user input = 3, run third menu option (display accuracy of training data)   
            case 3:
            	for (int i = 0; i < trainingInput.length; i++)
            	{
            		// Feed forward through the network using current weights/biases
            		neuralNetwork.feedForward(trainingInput[i]);
                	// Initialize all variables for total/correct guesses to 0
            		neuralNetwork.correct = 0; neuralNetwork.total = 0; neuralNetwork.correct0 = 0; neuralNetwork.total0 = 0; neuralNetwork.correct1 = 0; neuralNetwork.total1 = 0; neuralNetwork.correct2 = 0; neuralNetwork.total2 = 0; neuralNetwork.correct3 = 0; neuralNetwork.total3 = 0; neuralNetwork.correct4 = 0; neuralNetwork.total4 = 0; neuralNetwork.correct5 = 0; neuralNetwork.total5 = 0; neuralNetwork.correct6 = 0; neuralNetwork.total6 = 0; neuralNetwork.correct7 = 0; neuralNetwork.total7 = 0; neuralNetwork.correct8 = 0; neuralNetwork.total8 = 0; neuralNetwork.correct9 = 0; neuralNetwork.total9 = 0;          
            		// Guess the output, update guesses and totals
            		neuralNetwork.countGuesses(trainingOutput[i]);
            	}

                // Print the statistics 
                DecimalFormat percent = new DecimalFormat();
                double correctd = neuralNetwork.correct; double totald = neuralNetwork.total;
                System.out.println("0 = " + neuralNetwork.correct0 + "/" + neuralNetwork.total0 + "\t" + "1 = " + neuralNetwork.correct1 + "/" + neuralNetwork.total1 + "\t" + "2 = " + neuralNetwork.correct2 + "/" + neuralNetwork.total2 + "\t" + "3 = " + neuralNetwork.correct3 + "/" + neuralNetwork.total3 + "\t" + "4 = " + neuralNetwork.correct4 + "/" + neuralNetwork.total4 + "\t");
                System.out.println("5 = " + neuralNetwork.correct5 + "/" + neuralNetwork.total5 + "\t" + "6 = " + neuralNetwork.correct6 + "/" + neuralNetwork.total6 + "\t" + "7 = " + neuralNetwork.correct7 + "/" + neuralNetwork.total7 + "\t" + "8 = " + neuralNetwork.correct8 + "/" + neuralNetwork.total8 + "\t" + "9 = " + neuralNetwork.correct9 + "/" + neuralNetwork.total9 + "\t");
                System.out.println("Accuracy = " + neuralNetwork.correct + "/" + neuralNetwork.total + " = " + percent.format(correctd/totald*100) + "%\n");
            
            	break;

            // If user input = 4, run fourth menu option (display accuracy of testing data)   
            case 4:
            	for (int i = 0; i < testingInput.length; i++)
            	{
            		// Feed forward through the network using current weights/biases
            		neuralNetwork.feedForward(testingInput[i]);
                	// Initialize all variables for total/correct guesses to 0
            		neuralNetwork.correct = 0; neuralNetwork.total = 0; neuralNetwork.correct0 = 0; neuralNetwork.total0 = 0; neuralNetwork.correct1 = 0; neuralNetwork.total1 = 0; neuralNetwork.correct2 = 0; neuralNetwork.total2 = 0; neuralNetwork.correct3 = 0; neuralNetwork.total3 = 0; neuralNetwork.correct4 = 0; neuralNetwork.total4 = 0; neuralNetwork.correct5 = 0; neuralNetwork.total5 = 0; neuralNetwork.correct6 = 0; neuralNetwork.total6 = 0; neuralNetwork.correct7 = 0; neuralNetwork.total7 = 0; neuralNetwork.correct8 = 0; neuralNetwork.total8 = 0; neuralNetwork.correct9 = 0; neuralNetwork.total9 = 0;          
            		// Guess the output, update guesses and totals
            		neuralNetwork.countGuesses(testingOutput[i]);
            	}
                // Print the statistics 
                DecimalFormat percent1 = new DecimalFormat();
                double correctd1 = neuralNetwork.correct; double totald1 = neuralNetwork.total;
                System.out.println("0 = " + neuralNetwork.correct0 + "/" + neuralNetwork.total0 + "\t" + "1 = " + neuralNetwork.correct1 + "/" + neuralNetwork.total1 + "\t" + "2 = " + neuralNetwork.correct2 + "/" + neuralNetwork.total2 + "\t" + "3 = " + neuralNetwork.correct3 + "/" + neuralNetwork.total3 + "\t" + "4 = " + neuralNetwork.correct4 + "/" + neuralNetwork.total4 + "\t");
                System.out.println("5 = " + neuralNetwork.correct5 + "/" + neuralNetwork.total5 + "\t" + "6 = " + neuralNetwork.correct6 + "/" + neuralNetwork.total6 + "\t" + "7 = " + neuralNetwork.correct7 + "/" + neuralNetwork.total7 + "\t" + "8 = " + neuralNetwork.correct8 + "/" + neuralNetwork.total8 + "\t" + "9 = " + neuralNetwork.correct9 + "/" + neuralNetwork.total9 + "\t");
                System.out.println("Accuracy = " + neuralNetwork.correct + "/" + neuralNetwork.total + " = " + percent1.format(correctd1/totald1*100) + "%\n");
            	
            	break;
            	
            // If user input = 5, run fifth menu option (display testing images as ASCII art, show whether the guess was correct or incorrect) 
            case 5:
            	choice = 1;
            	while(choice == 1)
            	{
                	for (int i = 0; i < testingInput.length; i++)
                	{
                		// Feed forward through the network using current weights/biases
                		neuralNetwork.feedForward(testingInput[i]);
                		
                		// Create variable for storing correct guess
	            		double max = neuralNetwork.outputLayer[0];
	            		
	            		for(int k = 0; k < neuralNetwork.outputLayer.length; k++) 
	            		{        
	            			// Update correct guess by comparing each index until max is found
	            			if(max < neuralNetwork.outputLayer[k])
	            			{
	            				max = neuralNetwork.outputLayer[k];
	            			}       
	            		}
	            		
	            		// Store correct guess and network guess as variables using their index of the max guess
	            		int correctGuess = ArrayUtils.indexOf(testingOutput[i], 1);
	            		int networkGuess = ArrayUtils.indexOf(neuralNetwork.outputLayer, max);
	            		
	            		// Print statistics accordingly
	            		if(correctGuess == networkGuess)
	            		{
	            			System.out.println("Testing Case #" + i + ": \t Correct Classification = " + correctGuess + "\t Network Output = " + networkGuess + "\t Correct.\n");
	            			System.out.println(toASCII(testingInput[i]));
	            		}
	            		
	            		else
	            		{
	            			System.out.println("Testing Case #" + i + ": \t Correct Classification = " + correctGuess + "\t Network Output = " + networkGuess + "\t Incorrect.\n");
	            			System.out.println(toASCII(testingInput[i]));
	            		}
	            		
	            		// Continue iterating through testing set, or return to main menu based on user input 
	            		System.out.println("Enter 1 to continue. All other values will return to the main menu.\n");
	            		System.out.println("> ");
                        choice = userInput.nextInt();   
                    }
            	}
            	break;
            	
            // If user input = 6, run sixth menu option (display testing images as ASCII art, only display incorrect guesses.) 
            case 6:
            	choice = 1;
            	while(choice == 1)
            	{
                	for (int i = 0; i < testingInput.length; i++)
                	{
                		// Feed forward through the network using current weights/biases
                		neuralNetwork.feedForward(testingInput[i]);
                		
                		// Create variable for storing correct guess
	            		double max = neuralNetwork.outputLayer[0];
	            		
	            		for(int k = 0; k < neuralNetwork.outputLayer.length; k++) 
	            		{        
	            			// Update correct guess by comparing each index until max is found
	            			if(max < neuralNetwork.outputLayer[k])
	            			{
	            				max = neuralNetwork.outputLayer[k];
	            			}       
	            		}
	            		
	            		// Store correct guess and network guess as variables using their index of the max guess
	            		int correctGuess = ArrayUtils.indexOf(testingOutput[i], 1);
	            		int networkGuess = ArrayUtils.indexOf(neuralNetwork.outputLayer, max);
	            		
	            		// Print statistics accordingly
	            		if(correctGuess != networkGuess)
	            		{
	            			System.out.println("Testing Case #" + i + ": \t Correct Classification = " + correctGuess + "\t Network Output = " + networkGuess + "\t Incorrect.\n");
	            			System.out.println(toASCII(testingInput[i]));
	            			// Continue iterating through testing set, or return to main menu based on user input 
		            		System.out.println("Enter 1 to continue. All other values will return to the main menu.\n");
		            		System.out.println("> ");
	                        choice = userInput.nextInt(); 
	            		}
                    }
            	}
            	break;
            	
            // If user input = 7, run seventh menu option (save current weights to a file) 
            case 7:
            	// Create new file using File object
            	File output = new File("weights.txt");
        		if (output.createNewFile())
        		{
        			System.out.println("File created: " + output.getName() + "\n");
        		}
        		FileWriter write = new FileWriter("weights.txt");
        		
        		// Store weights as a simulated CSV, using commas to separate each value
        		StringBuilder builder = new StringBuilder();
        		for(int i = 0; i < neuralNetwork.weightsHidden.length; i++)
        		{
        			for(int j = 0; j < neuralNetwork.weightsHidden[i].length; j++)
        			{
        				builder.append(neuralNetwork.weightsHidden[i][j]+"");
        				if(j < neuralNetwork.weightsHidden.length - 1)
        					builder.append(",");
        			}
        			builder.append("\n");
        		}
        		for(int i = 0; i < neuralNetwork.weightsOutput.length; i++)
        		{
        			for(int j = 0; j < neuralNetwork.weightsOutput[i].length; j++)
        			{
        			   builder.append(neuralNetwork.weightsOutput[i][j]+"");
        			   if(j < neuralNetwork.weightsOutput.length - 1)
        				   builder.append(",");
        			}
        			builder.append("\n");
        		}
        		
        		// Write the entire string to the file created above
        		write.write(builder.toString());
        		write.close();
        		System.out.println("Weights saved successfully.\n");

            	break;
            	
            // If user input = 0, exit program.
            case 0:
            	System.out.println("Exiting Program...\n");
            	userInput.close();
                System.exit(0);  
                break;
            
            }
        }
    }
}