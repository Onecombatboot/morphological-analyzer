package com.drdo.issa.morphological_analyzer;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Paths;

@Service
public class OnnxEmbeddingService {

    private ZooModel<String, float[]> model;

    /** Overridable so the app can be run from anywhere, not only the source tree. */
    @Value("${gma.embedding.model-dir:models/all-MiniLM-L6-v2}")
    private String modelDir;

    @PostConstruct
    public void init() {
        try {
            String modelPath = Paths.get(modelDir).toAbsolutePath().toString();

            Criteria<String, float[]> criteria = Criteria.builder()
                    .setTypes(String.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optEngine("OnnxRuntime")
                    .optDevice(Device.cpu())
                    .optArgument("includeTokenTypes", "true")
                    .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                    .build();

            this.model = criteria.loadModel();
            System.out.println("✅ ONNX MiniLM Embedding Model loaded successfully natively in Java!");
        } catch (IOException | ModelException e) {
            System.err.println("❌ Failed to load ONNX MiniLM model: " + e.getMessage());
        }
    }

    // Converts text into a dense vector array
    public float[] getEmbedding(String text) {
        if (model == null) return new float[0];
        try (Predictor<String, float[]> predictor = model.newPredictor()) {
            return predictor.predict(text);
        } catch (TranslateException e) {
            System.err.println("Error generating embedding: " + e.getMessage());
            return new float[0];
        }
    }

    /** True if the model loaded. When false every embedding is empty and every similarity is 0. */
    public boolean isLoaded() {
        return model != null;
    }

    /**
     * Calculates the mathematical overlap (0.0 to 1.0) between two concepts.
     *
     * <p>Guards against a null, empty or mismatched vector rather than throwing: a failed embedding
     * returns {@code float[0]}, and the caller has no way to distinguish that from a genuine score
     * of zero, so the length is checked here where the information still exists.
     */
    public double cosineSimilarity(float[] vectorA, float[] vectorB) {
        if (vectorA == null || vectorB == null) return 0.0;
        if (vectorA.length == 0 || vectorA.length != vectorB.length) return 0.0;

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }
        if (normA == 0 || normB == 0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}