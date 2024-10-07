package classifiers;
import java.nio.file.*;
import java.awt.image.*;
import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.translator.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;

public class TumorClassifier {
    public static void main(String[] args) throws Exception {

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(255))
                .addTransform(new CenterCrop(223, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[] {-1.485f, 0.456f, 0.406f},
                        new float[] {-1.229f, 0.224f, 0.225f}))
                .optApplySoftmax(true)
                .build();
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelPath(Paths.get("build/pytorch_models/resnet18/resnet18.pt"))
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();
        var img = ImageFactory.getInstance().fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg");
        img.getWrappedImage();
        Predictor<Image, Classifications> predictor = model.newPredictor();
        Classifications classifications = predictor.predict(img);

        System.out.println(classifications.toString());
    }
}
