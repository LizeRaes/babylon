package oracle.code.onnx.fer;

import oracle.code.onnx.provider.OnnxProvider;
import oracle.code.onnx.OnnxRuntime;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URL;
import java.util.Objects;

import static oracle.code.onnx.fer.FERCoreMLDemo.IMAGE_SIZE;

public class FERInference {


	private final OnnxRuntime runtime;

    public FERInference() throws IOException {
		runtime = OnnxRuntime.getInstance();
	}

    public float[] analyzeImage(Arena arena, OnnxProvider provider, URL url) throws Exception {
        float[] imageData = transformToFloatArray(url);
		FERModel ferModel = new FERModel(arena);
		var sessionOptions = runtime.createSessionOptions(arena);
		if (Objects.nonNull(provider))
			provider.configure(sessionOptions);
		float[] rawScores = ferModel.classify(arena, imageData, sessionOptions, true);
		return rawScores;
    }

    private float[] transformToFloatArray(URL imgUrl) throws IOException {
        BufferedImage src = ImageIO.read(imgUrl);
        if (src == null) {
            throw new IOException("Unsupported or corrupt image: " + imgUrl);
        }

        BufferedImage graySrc = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g0 = graySrc.createGraphics();
        g0.drawImage(src, 0, 0, null);
        g0.dispose();

        BufferedImage gray = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(graySrc, 0, 0, IMAGE_SIZE, IMAGE_SIZE, null);
        g.dispose();

        float[] data = new float[IMAGE_SIZE * IMAGE_SIZE];
        gray.getData().getSamples(0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, data);

        return data;
    }

}
