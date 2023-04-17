package org.example;

import java.io.*;
import java.util.Arrays;

/**
 * @author hubert.cardot
 * @author Amine.ELACHAR
 * @author Abdallah.EL MAAZOUZI
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

        ///////////////////////////////////// Test 1: 1-PPV /////////////////////////////////////////
        System.out.println("************************ Test 1 - 1PPV *************************** ");
        // X is an exemple to classify (to take into data -test examples-)
        Double X[] = new Double[4] ;
        // Initialiser le vecteur
        X[0]=5.9; X[1]=3.0; X[2]=5.1; X[3]=1.8;

        // Initialiser la table distances
        Double[] distances = new Double[NbClasses*NbExLearning];
        ComputeDistances(X, distances);

        // Prédire La class pour notre exemple X en appelant la méthode findClass basée sur 1-PPV
        int predicted_class = findClass(distances);
        System.out.println(" ---> En utilisant 1PPv, la classe prédit pour l'exemple X est:" + predicted_class + "\n");

        ///////////////////////////////////// Test 2: k-PPV ///////////////////////////////////////
        System.out.println("************************ Test 2 - k-PPV *************************** ");
        // Prédire La class pour notre exemple X en appelant la méthode findClass basée sur k-PPV
        int predicted_class_kppv = findClassBasedKNearestNeighbours(distances,5);

        System.out.println(" ---> En utilisant k-PPv, la classe prédit pour l'exemple X est:" + predicted_class_kppv + "\n");


        ///////////////////////////////////// Evaluation /////////////////////////////////////////
        System.out.println("************************ Evaluation de notre modèle sur l'ensemble de test *************************** ");
        // Evaluer notre modèle en appelant la fonction evaluation
        evaluation();

        ///////////////////////////////////// Cross Validation ///////////////////////////
        // Afficher le résultat de l'accuracy donnée par la cross validation
        System.out.println("************************ La Cross Validation *************************** ");
        System.out.println(" - Accuracy given by cross-validation is: " + crossValidation___(5, 15) + "\n");
    }


    /***
     * Fonction qui calcule les distances euclidiennes entre un point de données d'entrée x et tous les exemples d'un ensemble d'apprentissage
     * @param x
     * @param distances
     */
    private static void ComputeDistances(Double x[], Double distances[]) {

        int index = 0;
        for(int i=0; i<NbClasses; i++){
            for(int j=0; j<NbExLearning; j++){
                Double distance = 0.0;
                for (int k = 0; k < NbFeatures; k++) {
                    distance +=  Math.pow(data[i][j][k] - x[k], 2);
                }
                distances[index] = Math.sqrt(distance);
                index++;
            }
        }
    }

    /***
     * Trouver la classe d'un point en utilisant la méthode du plus proche voisin 1-PPV
     * @param distances
     * @return
     */
    private static int findClass(Double distances[]) {
        double min = distances[0];
        int indice = 0;
        for (int i = 1; i < distances.length; i++) {
            if (distances[i] < min) {
                min = distances[i];
                indice = i;
            } else {
                continue;
            }
        }
        return indice / NbExLearning;
    }

    /***
     * Trouver la classe d'un point en utilisant la méthode des k plus proches voisins
     * @param distances
     * @param k
     * @return
     */
    private static int findClassBasedKNearestNeighbours(Double distances[], int k){

        // Intiliser des varibales
        int[] nearestClasses = new int[k];
        int[] indices = new int[k];

        // Trier notre table et sélectionner les k min distances.
        Double[] copy_distances = distances.clone();
        Arrays.sort(copy_distances);
        Double[] kDistances = Arrays.copyOfRange(copy_distances, 0,k);

        // Chercher les indices et les classes de k min distances séléctionnées
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

        // Compter le nombre d'apparition de chaque classe dans la table nearestClasses
        int[] count = new int[NbClasses];
        for (int c : nearestClasses) {
            count[c]++;
        }

        // Trouver la classe avec le nombre le plus élevé et retourner son index
        int maxIndex = 0;
        for (int i = 1; i < NbClasses; i++) {
            if (count[i] > count[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }


    /***
     * Fonction calculant l'accuracy en utlisant la Cross Validation
     * @param k
     * @param nFolds
     * @return
     */
    private static double crossValidation___(int k, int nFolds){
        // Initilisation
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

                    // Trouver les k voisins les plus proches
                    int predictedClass = findClassBasedKNearestNeighbours(distances, k);

                    // Vérifiez si la prédiction est correcte
                    if (predictedClass == trueClass) {
                        correct++;
                    }
                }
            }

            // Calculer la précision pour ce pli
            double foldAccuracy = correct / (foldSize*NbClasses);
            accuracy += foldAccuracy;
        }

        // Calculer la précision globale
        accuracy /= nFolds;
        return accuracy;
    }

    /***
     * Fonction ComputeDistance adaptée pour le calcule de la crossValidation
     * @param x
     * @param distances
     * @param testStart
     * @param testEnd
     */
    private static void ComputeDistances___(Double x[], Double distances[], int testStart, int testEnd) {
        // Initilisaiton
        int index = 0;

        // calculer la distance euclidienne entre x et chaque élément de l'ensemble d'apprentissage
        for(int i=0; i<NbClasses; i++){
            for(int j=0; j<NbExLearning; j++){
                if (j<testStart || j >= testEnd) {
                    Double distance = 0.0;
                    for (int k = 0; k < NbFeatures; k++) {
                        distance += Math.pow(data[i][j][k] - x[k], 2);
                    }
                    // Ajouter au tableau des distances
                    distances[index] = Math.sqrt(distance);
                    index++;
                }
            }
        }
    }

    /***
     * Evoluer notre algorithme en calculant les deux indicateurs Matrice de confusion & le Taux de reconnaissance
     */
    private static void evaluation(){
        // initialisation de nos variables
        int NbExTesting = NbEx - NbExLearning;
        int prediction[] = new int[NbClasses*NbExTesting];
        int[][] confusionMatrix = new int[NbClasses][NbClasses];
        int correctPredictions = 0;
        int index = 0;

        // Calculation de la Matrice de confusion & le Taux de reconnaissance
        for(int i=0; i<NbClasses; i++){
            for(int j=NbExLearning; j<NbEx; j++){
                Double X[] = data[i][j];
                Double distances[] = new Double[NbClasses*NbExTesting];

                // Calcule des distances pour l'exemple X
                ComputeDistances(X, distances);

                // Faire la prédiction pour l'exmeple X en se basant sur 30 les plus proches K-PPV ave
                int X_predicted_class = findClassBasedKNearestNeighbours(distances, 10);
                //Faire la prédiction pour l'exmeple X en se basant le plus proche voisin 1-PPv
                //int X_predicted_class = findClass(distances);
                prediction[index] = X_predicted_class;
                int real_class = i;

                confusionMatrix[real_class][X_predicted_class]++;
                index++;

                if (X_predicted_class == i) {
                    correctPredictions++;
                }
            }
        }

        // Matrice de confusion
        System.out.println("Matrice de confusion :");
        for (int i = 0; i < NbClasses; i++) {
            for (int j = 0; j < NbClasses; j++) {
                System.out.print(confusionMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        // Taux de reconnaissance
        double recognitionRate = (double) correctPredictions / (NbClasses * (NbEx - NbExLearning));
        System.out.println("Le taux de reconnaisance: " + recognitionRate );
    }


    //——-Reading data from iris.data file
	//1 line -> 1 exemple
	//50 first lines are 50 exemples of class 0, next 50 of class 1 and 50 of class 2
    private static void ReadFile() {

        String line, subPart;
        int classe=0, n=0;
        try {
             BufferedReader fic=new BufferedReader(new FileReader("src/main/java/org/example/iris.data"));
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
