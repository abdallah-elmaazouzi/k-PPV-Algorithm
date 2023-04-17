package org.example;

import java.io.*;
import java.util.Arrays;

/**
 * @author hubert.cardot
 */
public class kPPV {
	//General variables for dealing with Iris data (UCI repository)
	// NbEx: number of data per class in dataset
	// NbClasses: Number of classes to recognize
	// NbFeatures: Dimensionality of dataset
	// NbExLearning: Number of exemples per class used for learning (there are the first ones in data storage for each class)

    static int NbEx=50, NbClasses=3, NbFeatures=4, NbExLearning=25;
    static Double data[][][] = new Double[NbClasses][NbEx][NbFeatures];//there are 50*3 exemples at all. All have 4 features

    public static void main(String[] args) {
        System.out.println("Starting kPPV");
        ReadFile();

        //X is an exemple to classify (to take into data -test examples-)
        Double X[] = new Double[NbFeatures];
        // distances: table to store all distances between the given exemple X and all exemples in learning set, using ComputeDistances
        Double distances[] = new Double[NbClasses*NbExLearning];

        //To be done
        X[0] = data[0][27][0];X[1] = data[0][27][1];X[2] = data[0][27][2];X[3] = data[0][27][3];
        ComputeDistances(X, distances);
        //int X_predicted_class = findClass(distances);
        //System.out.println(X_predicted_class);

        evaluation();
        //findClass4(distances, 3);
        System.out.println("Accuracy  with cross validation is: " + crossValidation___(5, 10));

    }


    /*
    * calculer les distances euclidiennes entre un point de données d'entrée x et tous les exemples d'un ensemble d'apprentissage
    * */
    private static void ComputeDistances(Double x[], Double distances[]) {
        //---compute the distance between an input data x to test and all examples in training set (in data)
        int index = 0;
        for(int i=0; i<NbClasses; i++){
            for(int j=0; j<NbExLearning; j++){
                // calculate the euclidian distance  between x and each element of training set
                Double distance = 0.0;
                for (int k = 0; k < NbFeatures; k++) {
                    distance +=  Math.pow(data[i][j][k] - x[k], 2);
                }
                // add to distances array
                distances[index] = Math.sqrt(distance);
                index++;
            }
        }
    }

    /*
    *  déterminer la classe la plus proche en se basant sur les distances calculées entre une instance de test et toutes les instances de l'ensemble d'apprentissage.
    * */
    private static int findClass1(Double distances[], int k) {
        // initialize arrays to hold nearest classes and indices of k nearest neighbors
        Integer[] indices = new Integer[k];
        int[] nearestClasses = new int[k];

        // initialize nearestClasses and indices arrays with the first k values
        for (int i = 0; i < k; i++) {
            indices[i] = i;
            nearestClasses[i] = indices[i] / NbExLearning;
        }

        // find the k nearest neighbors by updating the indices array and sorting it
        // based on distance

        for (int i = k; i < distances.length; i++) {
            double dist = distances[i];
            if (dist < distances[indices[k - 1]]) {
                indices[k - 1] = i;
                // sort indices array based on distance using a lambda expression
                Arrays.sort(indices, (a, b) -> Double.compare(distances[a], distances[b]));
                // update nearestClasses array based on new indices
                for (int j = 0; j < k; j++) {
                    nearestClasses[j] = indices[j] / NbExLearning;
                }
            }
        }
        // count the number of occurrences of each class in nearestClasses array
        int[] count = new int[NbClasses];
        for (int c : nearestClasses) {
            count[c]++;
        }
        // find the class with the highest count and return its index
        int maxIndex = 0;
        for (int i = 1; i < NbClasses; i++) {
            if (count[i] > count[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static int findClass4(Double distances[], int k){

        // on initialise nos variables
        int[] nearestClasses = new int[k];
        int[] indices = new int[k];

        // trier notre table et sélectionner les k min distances.
        Double[] copy_distances = Arrays.copyOf(distances, 75);
        Arrays.sort(copy_distances);
        Double[] kDistances = Arrays.copyOfRange(copy_distances, 0,k);

        int index=0;
        for(Double v:kDistances){
            for(int i=0; i<distances.length; i++){
                if(distances[i].equals(v)){
                    indices[index] = i;
                    nearestClasses[index]=i/NbExLearning;
                    index++;
                    break;
                }
            }
            if(nearestClasses.length==3){
                break;
            }
        }

        // count the number of occurrences of each class in nearestClasses array
        int[] count = new int[NbClasses];
        for (int c : nearestClasses) {
            count[c]++;
        }

        // find the class with the highest count and return its index
        int maxIndex = 0;
        for (int i = 1; i < NbClasses; i++) {
            if (count[i] > count[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private static int findClass(Double distances[]){
        // initilization
        Double min_distance = distances[0];
        int nearest_index = -1;

        // searching the min distance in the array
        for(int i=1; i<distances.length; i++){
            if(min_distance > distances[i]){
                min_distance = distances[i];
                nearest_index = i;
            }
        }
        return nearest_index/NbExLearning; //(nearest_index >= 0 && nearest_index < 25) ? 0 : (nearest_index >= 25 && nearest_index < 50) ? 1 : 2;
    }

    private static double crossValidation___(int k, int nFolds){
        double accuracy = 0;
        int foldSize = NbExLearning/nFolds;

        for(int fold=0; fold<nFolds; fold++){
            int testStart = fold*foldSize;
            int testEnd = testStart  + foldSize;
            double correct=0;


            for(int i=0; i<NbClasses; i++) {
                for (int j = testStart; j < testEnd; j++) {
                    Double[] example = data[i][j];
                    int trueClass = i;

                    Double[] distances = new Double[NbClasses * (NbExLearning - foldSize)];
                    ComputeDistances___(example, distances, testStart, testEnd);

                    // Find the k nearest neighbors
                    int predictedClass = findClass___(distances, k);

                    // Check if prediction is correct
                    if (predictedClass == trueClass) {
                        correct++;
                    }
                }
            }

            // Compute accuracy for this fold
            double foldAccuracy = correct / (foldSize*NbClasses);
            accuracy += foldAccuracy;
        }

        // Compute overall accuracy
        accuracy /= nFolds;
        return accuracy;
    }

    private static void ComputeDistances___(Double x[], Double distances[], int testStart, int testEnd) {
        //---compute the distance between an input data x to test and all examples in training set (in data)
        int index = 0;
        for(int i=0; i<NbClasses; i++){
            for(int j=0; j<NbExLearning; j++){
                if (j<testStart || j >= testEnd) {
                    // calculate the euclidian distance  between x and each element of training set
                    Double distance = 0.0;
                    for (int k = 0; k < NbFeatures; k++) {
                        distance += Math.pow(data[i][j][k] - x[k], 2);
                    }
                    // add to distances array
                    distances[index] = Math.sqrt(distance);
                    index++;
                }
            }
        }
    }
    private static int findClass___(Double distances[], int k){

        // on initialise nos variables
        int[] nearestClasses = new int[k];
        int[] indices = new int[k];

        // trier notre table et sélectionner les k min distances.
        Double[] copy_distances = Arrays.copyOf(distances, 60);
        Arrays.sort(copy_distances);
        Double[] kDistances = Arrays.copyOfRange(copy_distances, 0,k);

        int index=0;
        for(Double v:kDistances){
            for(int i=0; i<distances.length; i++){
                if(distances[i].equals(v)){
                    indices[index] = i;
                    nearestClasses[index]=i/NbExLearning;
                    index++;
                    break;
                }
            }
            if(nearestClasses.length==3){
                break;
            }
        }

        // count the number of occurrences of each class in nearestClasses array
        int[] count = new int[NbClasses];
        for (int c : nearestClasses) {
            count[c]++;
        }

        // find the class with the highest count and return its index
        int maxIndex = 0;
        for (int i = 1; i < NbClasses; i++) {
            if (count[i] > count[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }
    public static double crossValidation_(int k, int nFolds) {
        double accuracy = 0;
        int foldSize = NbEx / nFolds;

        // Randomly shuffle the data
        //shuffleData();

        // Loop over each fold
        for (int fold = 0; fold < nFolds; fold++) {
            int testStart = fold * foldSize;
            int testEnd = testStart + foldSize;
            double correct = 0;

            // Loop over each example in the test set
            for (int i = testStart; i < testEnd; i++) {
                // Get the example and its true class
                Double[] example = data[i / NbExLearning][i % NbExLearning];
                int trueClass = i / NbExLearning;

                // Compute distances to all training examples
                Double[] distances = new Double[NbClasses * NbExLearning];
                ComputeDistances(example, distances);

                // Find the k nearest neighbors
                int predictedClass = findClass4(distances, k);

                // Check if prediction is correct
                if (predictedClass == trueClass) {
                    correct++;
                }
            }

            // Compute accuracy for this fold
            double foldAccuracy = correct / foldSize;
            accuracy += foldAccuracy;
        }

        // Compute overall accuracy
        accuracy /= nFolds;
        return accuracy;
    }

    /*
    * Evoluer notre algorithm en calculant les deux indicateurs Matrice de confusion & le Taux de reconnaissance */
    private static void evaluation(){
        // initialization
        int NbExTesting = NbEx - NbExLearning;
        int prediction[] = new int[NbClasses*NbExTesting];

        int[][] confusionMatrix = new int[NbClasses][NbClasses];
        int correctPredictions = 0;

        int index = 0;
        for(int i=0; i<NbClasses; i++){
            for(int j=NbExLearning; j<NbEx; j++){
                Double X[] = data[i][j];
                Double distances[] = new Double[NbClasses*NbExTesting];
                ComputeDistances(X, distances);
                int X_predicted_class = findClass4(distances, 30);
                prediction[index] = X_predicted_class;
                int real_class = i;

                confusionMatrix[real_class][X_predicted_class]++;
                index++;

                if (X_predicted_class == i) {
                    correctPredictions++;
                }
            }
        }

        System.out.println("Matrice de confusion :");
        for (int i = 0; i < NbClasses; i++) {
            for (int j = 0; j < NbClasses; j++) {
                System.out.print(confusionMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        // Calcul du taux de reconnaissance
        double recognitionRate = (double) correctPredictions / (NbClasses * (NbEx - NbExLearning));
        /*

        double recognitionRate = 0.0;
        for (int i = 0; i < NbClasses; i++) {
            recognitionRate += confusionMatrix[i][i];
        }
        recognitionRate /= NbClasses*NbExTesting;

         */
        System.out.println("recognitionRate: " + recognitionRate );
    }

    //——-Reading data from iris.data file
	//1 line -> 1 exemple
	//50 first lines are 50 exemples of class 0, next 50 of class 1 and 50 of class 2
    private static void ReadFile() {

        String line, subPart;
        int classe=0, n=0;
        try {
             BufferedReader fic=new BufferedReader(new FileReader("C:\\Users\\HomePC\\OneDrive\\Documents\\BDMA S1\\ReconaissanceImg\\TP\\Original\\kPPV\\iris.data"));
             while ((line=fic.readLine())!=null) {
                for(int i=0;i<NbFeatures;i++) {
                    subPart = line.substring(i*NbFeatures, i*NbFeatures+3);
                    data[classe][n][i] = Double.parseDouble(subPart);
                    //System.out.println(data[classe][n][i]+" "+classe+" "+n);
                }
                if (++n==NbEx) { n=0; classe++; }
             }
        }
        catch (Exception e) { System.out.println(e.toString()); }
    }

} //-------------------End of class kPPV-------------------------
