package org.github.kingster;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class HuggingFaceIntentClassifier {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceIntentClassifier.class);

    // Define your intent labels in the same order as they were trained
    private static final List<String> labels = Arrays.asList(
            "goodbye",
            "greet",
            "others",
            "thank_you"
    );
    private static final double CONFIDENCE_THRESHOLD = 0.7;
    private static final int MAX_SEQ_LENGTH = 512;

    private final Predictor<String, Classifications> predictor;

    public HuggingFaceIntentClassifier(Path modelPath, Path tokenizerPath) throws MalformedModelException, IOException, ModelNotFoundException {

        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath, Collections.singletonMap("padding", "false"));

        // Build Criteria with our custom translator
        Criteria<String, Classifications> criteria = Criteria.builder()
                .optApplication(Application.NLP.TEXT_CLASSIFICATION)
                .setTypes(String.class, Classifications.class)
                .optModelPath(modelPath)
                .optTranslator(new BertEmbedTranslator(tokenizer, MAX_SEQ_LENGTH))
                .optEngine("OnnxRuntime")
                .build();

        // Load the model and create the predictor
        ZooModel<String, Classifications> model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    public String classify(String text) throws TranslateException {
        Classifications classifications = getEmbedding(text);
        // Find max probability and predicted label
        Classifications.Classification classified = classifications.topK(1).getFirst();
        double probability = classified.getProbability();

        if (probability < CONFIDENCE_THRESHOLD) {
            return "others";
        }
        return classified.getClassName();
    }


    public Classifications getEmbedding(String text) throws TranslateException {
        return predictor.predict(text);
    }

    /**
     * Custom translator that tokenizes the text and gets the classification
     */
    private static class BertEmbedTranslator implements Translator<String, Classifications> {
        private final HuggingFaceTokenizer tokenizer;
        private final int maxSeqLength;

        public BertEmbedTranslator(HuggingFaceTokenizer tokenizer, int maxSequenceLength) {
            this.tokenizer = tokenizer;
            this.maxSeqLength = maxSequenceLength;
        }

        private String toText(List<String> tokens) {
            String text = this.tokenizer.buildSentence(tokens);
            List<String> tokenized = this.tokenizer.tokenize(text);
            List<String> tokenizedWithoutSpecialTokens = new LinkedList(tokenized);
            tokenizedWithoutSpecialTokens.remove(0);
            tokenizedWithoutSpecialTokens.remove(tokenizedWithoutSpecialTokens.size() - 1);
            return tokenizedWithoutSpecialTokens.equals(tokens) ? text : String.join("", tokens);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {

            List<String> tokens = tokenizer.tokenize(input);
            logger.debug("tokens: {}", tokens);

            // Check for unknown tokens [UNK]
            if (tokens.contains("[UNK]")) {
                logger.warn("WARNING: [UNK] token generated for this input!");
            }

            // Check if truncation occurred at 512 tokens
            if (tokens.size() > maxSeqLength) {
                tokens = tokens.subList(0, maxSeqLength);
                logger.warn("DEBUG - Truncated to the first {} tokens.", maxSeqLength);
            }

            // Convert tokens to IDs
            Encoding encoding = tokenizer.encode(this.toText(tokens), true, false);
            long[] inputIds = encoding.getIds() ;

            long[] attentionMaskArr = new long[tokens.size()];
            Arrays.fill(attentionMaskArr, 1);

            // Create attention_mask and token_type_ids
            long[] attentionMask = new long[inputIds.length];
            Arrays.fill(attentionMask, 1);
            long[] tokenTypeIds = new long[inputIds.length];
            Arrays.fill(tokenTypeIds, tokens.size(), tokenTypeIds.length, 1);

            // Create NDArrays
            NDArray inputIdsArr = ctx.getNDManager().create(inputIds);
            NDArray attentionArr = ctx.getNDManager().create(attentionMask);
            NDArray tokenTypeArr = ctx.getNDManager().create(tokenTypeIds);

            // Assign names
            inputIdsArr.setName("input_ids");
            attentionArr.setName("attention_mask");
            tokenTypeArr.setName("token_type_ids");

            return new NDList(inputIdsArr, attentionArr, tokenTypeArr);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray raw = list.singletonOrThrow();
            NDArray computed = raw.softmax(0); //.exp().div(raw.exp().sum(new int[] {0}, true)); // apply softmax
            return new Classifications(labels, computed);
        }
    }

}
