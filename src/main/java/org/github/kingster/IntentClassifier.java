package org.github.kingster;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IntentClassifier {

    private static Logger logger = LoggerFactory.getLogger(IntentClassifier.class);

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

    public IntentClassifier(Path modelPath, Path vocabPath) throws MalformedModelException, IOException, ModelNotFoundException {

        // Load the vocabulary
        Vocabulary vocab = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(vocabPath)
                .optUnknownToken("[UNK]")
                .build();

        // Instantiate a simple BERT tokenizer
        BertFullTokenizer tokenizer = new BertFullTokenizer(vocab, true);

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
        private final BertFullTokenizer tokenizer;
        private final Vocabulary vocab;
        private final int maxSeqLength;

        public BertEmbedTranslator(BertFullTokenizer tokenizer, int maxSequenceLength) {
            this.tokenizer = tokenizer;
            this.vocab = tokenizer.getVocabulary();
            this.maxSeqLength = maxSequenceLength;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {

            List<String> tokens = new ArrayList<>();
            // BERT embedding convention "[CLS] Your Sentence [SEP]"
            // Reference https://docs.djl.ai/master/docs/demos/jupyter/rank_classification_using_BERT_on_Amazon_Review.html
            tokens.add("[CLS]");
            tokens.addAll(tokenizer.tokenize(input));
            tokens.add("[SEP]");

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
            long[] inputIds = tokens.stream().mapToLong(vocab::getIndex).toArray();

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
