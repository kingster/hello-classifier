package org.github.kingster;

import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.StringPair;

import java.nio.file.Path;
import java.nio.file.Paths;

public class ReRanker {


//    from transformers import AutoModelForSequenceClassification, AutoTokenizer
//
//            tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
//    model_ort = ORTModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base', file_name="onnx/model.onnx")
//
//            # Sentences we want sentence embeddings for
//    pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
//
//            # Tokenize sentences
//    encoded_input = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
//    scores_ort = model_ort(**encoded_input, return_dict=True).logits.view(-1, ).float()
//
//# scores and scores_ort are identical

    public static void main(String[] args) throws Exception {

        // Get resources from the classpath root
        Path modelPath = Paths.get("/Volumes/Workspace/code/ai/hello-classifier/src/main/resources/bge-reranker-base");

        StringPair[] records = new StringPair[]{
                new StringPair("what is panda?", "Today is a sunny day"),
                new StringPair("what is panda?", "The tiger (Panthera tigris) is a member of the genus Panthera and the largest living cat species native to Asia."),
                new StringPair("what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."),
        };


        Criteria<StringPair, float[]> criteria = Criteria.builder()
                .setTypes(StringPair.class, float[].class)
                .optEngine("OnnxRuntime")
                .optModelPath(modelPath)
                .optArgument("reranking", "true")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .build();

        ZooModel<StringPair, float[]> model = criteria.loadModel();
        Predictor<StringPair, float[]> predictor = model.newPredictor();


        for (StringPair record : records) {
            float[] res = predictor.predict(record);
            System.out.println(res[0]);
        }


    }


}
