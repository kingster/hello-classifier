package org.github.kingster;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Hello world!
 *
 */
public class App  {

    public static void main(String[] args) throws Exception {

        // Get resources from the classpath root
        URL modelURL = App.class.getClassLoader().getResource("intent_model_onnx/model.onnx");
        if (modelURL == null) {
            throw new RuntimeException("Could not find model.onnx in resources");
        }
        Path modelPath = Paths.get(modelURL.toURI());

        URL vocabURL = App.class.getClassLoader().getResource("intent_model_onnx/vocab.txt");
        if (vocabURL == null) {
            throw new RuntimeException("Could not find vocab.txt in resources");
        }
        Path vocabPath = Paths.get(vocabURL.toURI());

        URL tokenizerURL = App.class.getClassLoader().getResource("intent_model_onnx/tokenizer.json");
        if (tokenizerURL == null) {
            throw new RuntimeException("Could not find tokenizer.json in resources");
        }
        Path tokenizerPath = Paths.get(tokenizerURL.toURI());

        // Instantiate the embedding engine
        IntentClassifier engine1 = new IntentClassifier(modelPath, vocabPath);
        HuggingFaceIntentClassifier engine2 = new HuggingFaceIntentClassifier(modelPath, tokenizerPath);

        // Test some examples
        String[] testPhrases = {
                "hiya!",
                "hi there!",
                "tell me a joke",
                "thank you so much",
                "what is the meaning of life?",
                "can you help me with quantum physics?",
                "random gibberish text",
                "some random shit"
        };

        for (String phrase : testPhrases) {
            String intent1 = engine1.classify(phrase);
            String intent2 = engine2.classify(phrase);
            System.out.println("\nText: " + phrase);
            System.out.println("Predicted Intent: " + intent1 );
            System.out.println("Predicted Intent: " + intent2 );

        }
    }
}
