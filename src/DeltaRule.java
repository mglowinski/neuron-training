import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class DeltaRule {

    private static final int NEURON_INPUTS_NUMBER = 3;
    private static final int TRAINING_EPOCHS_NUMBER = 100;
    private static final int TRAINING_PATTERNS_NUMBER = 1;

    private static final double TRAINING_STEP = 0.05;

    private static final double minIntervalWeights = -2;
    private static final double maxIntervalWeights = 2;

    private static final double minIntervalInputs = -2;
    private static final double maxIntervalInputs = 2;

    private static final double minIntervalOutputs = -2;
    private static final double maxIntervalOutputs = 2;

    private double[][] inputs;
    private double[] weights;
    private double[] outputs;

    public static void main(String[] args) {
        DeltaRule deltaRule = new DeltaRule();

        deltaRule.initialInputs();
        deltaRule.initialWeights();
        deltaRule.initialOutputs();

        deltaRule.run();
    }

    private void initialInputs() {
        inputs = new double[TRAINING_PATTERNS_NUMBER][NEURON_INPUTS_NUMBER];

        for (int i = 0; i < TRAINING_PATTERNS_NUMBER; i++) {
            for (int j = 0; j < NEURON_INPUTS_NUMBER; j++) {
                inputs[i][j] = getRandomDoubleValue(minIntervalInputs, maxIntervalInputs);
            }
        }
    }

    private void initialWeights() {
        weights = new double[NEURON_INPUTS_NUMBER];

        for (int i = 0; i < NEURON_INPUTS_NUMBER; i++) {
            weights[i] = getRandomDoubleValue(minIntervalWeights, maxIntervalWeights);
        }
    }

    private void initialOutputs() {
        outputs = new double[TRAINING_PATTERNS_NUMBER];

        for (int i = 0; i < TRAINING_PATTERNS_NUMBER; i++) {
            outputs[i] = getRandomDoubleValue(minIntervalOutputs, maxIntervalOutputs);
        }
    }

    private double getRandomDoubleValue(double min, double max) {
        return ThreadLocalRandom.current().nextDouble(min, max + 1);
    }

    private void run() {
        for (int i = 0; i < TRAINING_EPOCHS_NUMBER; i++) {
            System.out.println("***Beginning Epoch #" + (i + 1) + "***");

            prepareEpoch();
        }
    }

    private void prepareEpoch() {
        for (int i = 0; i < TRAINING_PATTERNS_NUMBER; i++) {
            presentPattern(i);
        }
    }

    private void presentPattern(int patternNumber) {
        double[] currentTrainingPattern = inputs[patternNumber];

        System.out.println("Current Training Pattern" + Arrays.toString(currentTrainingPattern));

        double actualSum = sumElements(currentTrainingPattern);
        double error = calculateError(actualSum, outputs[patternNumber]);

        for (int i = 0; i < weights.length; i++) {
            double delta = trainingFunction(currentTrainingPattern[i], error);
            weights[i] += delta;
        }

        System.out.println("actual=" + actualSum);
        System.out.println("anticipated=" + outputs[patternNumber]);
        System.out.println("error=" + error);

        System.out.println("Current Weights" + Arrays.toString(weights));
    }

    private double sumElements(double[] currentTrainingPattern) {
        double sum = 0;

        for (int i = 0; i < currentTrainingPattern.length; i++) {
            sum += currentTrainingPattern[i] * weights[i];
        }

        return sum;
    }

    private double calculateError(double actual, double anticipated) {
        return (anticipated - actual);
    }

    private double trainingFunction(double input, double error) {
        return TRAINING_STEP * input * error;
    }

}
