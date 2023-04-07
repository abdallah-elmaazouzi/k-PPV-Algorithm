package org.example;

import java.io.*;

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
        int X_predicted_class = findClass(distances);
        System.out.println(X_predicted_class);

        evaluation();

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
        return (nearest_index >= 0 && nearest_index < 25) ? 0 : (nearest_index >= 25 && nearest_index < 50) ? 1 : 2;
    }

    /*
    * Evoluer notre algorithm en calculant les deux indicateurs Matrice de confusion & le Taux de reconnaissance */
    private static void evaluation(){
        // initialization
        int NbExTesting = NbEx - NbExLearning;
        int prediction[] = new int[NbClasses*NbExTesting];

        int[][] confusionMatrix = new int[NbClasses][NbClasses];

        int index = 0;
        for(int i=0; i<NbClasses; i++){
            for(int j=NbExLearning; j<NbEx; j++){
                Double X[] = data[i][j];
                Double distances[] = new Double[NbClasses*NbExTesting];

                /*for (int k = 0; k < NbFeatures; k++) {
                    X[k] =  data[i][j][k];
                }*/

                ComputeDistances(X, distances);
                int X_predicted_class = findClass(distances);
                prediction[index++] = X_predicted_class;
                int real_class = i;

                confusionMatrix[real_class][X_predicted_class]++;
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
        double recognitionRate = 0.0;
        for (int i = 0; i < NbClasses; i++) {
            recognitionRate += confusionMatrix[i][i];
        }
        recognitionRate /= NbClasses*NbExTesting;
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
