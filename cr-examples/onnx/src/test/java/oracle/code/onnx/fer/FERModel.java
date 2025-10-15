package oracle.code.onnx.fer;

import jdk.incubator.code.CodeReflection;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.genai.TensorDataStream;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.net.URL;
import java.util.Objects;

import static oracle.code.onnx.OnnxOperators.*;
import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.fer.FERCoreMLDemo.IMAGE_SIZE;

public class FERModel {

	// Weights (constant inputs)
	final Tensor<Float> parameter1693;
	final Tensor<Float> parameter1403;
	final Tensor<Float> parameter1367;
	final Tensor<Float> parameter695;
	final Tensor<Float> parameter675;
	final Tensor<Float> parameter655;
	final Tensor<Float> parameter615;
	final Tensor<Float> parameter595;
	final Tensor<Float> parameter575;
	final Tensor<Float> parameter83;
	final Tensor<Float> parameter63;
	final Tensor<Float> parameter23;
	final Tensor<Float> parameter3;
	final Tensor<Float> constant339;
	final Tensor<Float> constant343;
	final Tensor<Float> parameter4;
	final Tensor<Float> parameter24;
	final Tensor<Float> parameter64;
	final Tensor<Float> parameter84;
	final Tensor<Float> parameter576;
	final Tensor<Float> parameter596;
	final Tensor<Float> parameter616;
	final Tensor<Float> parameter656;
	final Tensor<Float> parameter676;
	final Tensor<Float> parameter696;
	final Tensor<Long> dropout612_reshape0_shape;
	final Tensor<Long> parameter1367_reshape1_shape;
	final Tensor<Float> parameter1368;
	final Tensor<Float> parameter1404;
	final Tensor<Float> parameter1694;

	public FERModel(Arena arena) throws IOException {
		URL resource = Objects.requireNonNull(FERModel.class.getResource("emotion-ferplus-8.onnx.data"));
		var modelData = new TensorDataStream(arena, resource.getPath());
		parameter1693 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024, 8);
		parameter1403 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024, 1024);
		parameter1367 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 4, 4, 1024);
		parameter695 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
		parameter675 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
		parameter655 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
		parameter615 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
		parameter595 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
		parameter575 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 128, 3, 3);
		parameter83 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 128, 3, 3);
		parameter63 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 64, 3, 3);
		parameter23 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 64, 3, 3);
		parameter3 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 3, 3);
		constant339 = Tensor.ofScalar(127.5f);
		constant343 = Tensor.ofScalar(255.0f);
		parameter4 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 1);
		parameter24 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 1);
		parameter64 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 1, 1);
		parameter84 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 1, 1);
		parameter576 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		parameter596 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		parameter616 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		parameter656 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		parameter676 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		parameter696 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
		dropout612_reshape0_shape = Tensor.ofShape(new long[]{2}, 1, 4096);
		parameter1367_reshape1_shape = Tensor.ofShape(new long[]{2}, 4096, 1024);
		parameter1368 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024);
		parameter1404 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024);
		parameter1694 = modelData.nextTensor(Tensor.ElementType.FLOAT, 8);
	}

	@CodeReflection
	public Tensor<Float> cntkGraph(Tensor<Float> input3) {
		Tensor<Float> parameter1367_reshape1 = Reshape(parameter1367, parameter1367_reshape1_shape, empty());
		Tensor<Float> minus340 = Sub(input3, constant339);
		Tensor<Float> block352 = Div(minus340, constant343);
		Tensor<Float> convolution362 = Conv(block352, parameter3, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus364 = Add(convolution362, parameter4);
		Tensor<Float> reLU366 = Relu(plus364);
		Tensor<Float> convolution380 = Conv(reLU366, parameter23, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus382 = Add(convolution380, parameter24);
		Tensor<Float> reLU384 = Relu(plus382);
		var pooling398 = MaxPool(reLU384, of(new long[] {0l, 0l, 0l, 0l}), empty(), of("NOTSET"), empty(), empty(), of(new long[] {2l, 2l}), new long[] {2l, 2l});
		var dropout408 = Dropout(pooling398.Y(), empty(), empty(), empty());
		Tensor<Float> convolution418 = Conv(dropout408.output(), parameter63, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus420 = Add(convolution418, parameter64);
		Tensor<Float> reLU422 = Relu(plus420);
		Tensor<Float> convolution436 = Conv(reLU422, parameter83, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus438 = Add(convolution436, parameter84);
		Tensor<Float> reLU440 = Relu(plus438);
		var pooling454 = MaxPool(reLU440, of(new long[] {0l, 0l, 0l, 0l}), empty(), of("NOTSET"), empty(), empty(), of(new long[] {2l, 2l}), new long[] {2l, 2l});
		var dropout464 = Dropout(pooling454.Y(), empty(), empty(), empty());
		Tensor<Float> convolution474 = Conv(dropout464.output(), parameter575, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus476 = Add(convolution474, parameter576);
		Tensor<Float> reLU478 = Relu(plus476);
		Tensor<Float> convolution492 = Conv(reLU478, parameter595, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus494 = Add(convolution492, parameter596);
		Tensor<Float> reLU496 = Relu(plus494);
		Tensor<Float> convolution510 = Conv(reLU496, parameter615, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus512 = Add(convolution510, parameter616);
		Tensor<Float> reLU514 = Relu(plus512);
		var pooling528 = MaxPool(reLU514, of(new long[] {0l, 0l, 0l, 0l}), empty(), of("NOTSET"), empty(), empty(), of(new long[] {2l, 2l}), new long[] {2l, 2l});
		var dropout538 = Dropout(pooling528.Y(), empty(), empty(), empty());
		Tensor<Float> convolution548 = Conv(dropout538.output(), parameter655, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus550 = Add(convolution548, parameter656);
		Tensor<Float> reLU552 = Relu(plus550);
		Tensor<Float> convolution566 = Conv(reLU552, parameter675, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus568 = Add(convolution566, parameter676);
		Tensor<Float> reLU570 = Relu(plus568);
		Tensor<Float> convolution584 = Conv(reLU570, parameter695, empty(), empty(), of(new long[] {1l, 1l}), of("SAME_UPPER"), of(new long[] {1l, 1l}), of(1l), of(new long[] {3l, 3l}));
		Tensor<Float> plus586 = Add(convolution584, parameter696);
		Tensor<Float> reLU588 = Relu(plus586);
		var pooling602 = MaxPool(reLU588, of(new long[] {0l, 0l, 0l, 0l}), empty(), of("NOTSET"), empty(), empty(), of(new long[] {2l, 2l}), new long[] {2l, 2l});
		var dropout612 = Dropout(pooling602.Y(), empty(), empty(), empty());
		Tensor<Float> dropout612_reshape0 = Reshape(dropout612.output(), dropout612_reshape0_shape, empty());
		Tensor<Float> times622 = MatMul(dropout612_reshape0, parameter1367_reshape1);
		Tensor<Float> plus624 = Add(times622, parameter1368);
		Tensor<Float> reLU636 = Relu(plus624);
		var dropout646 = Dropout(reLU636, empty(), empty(), empty());
		Tensor<Float> times656 = MatMul(dropout646.output(), parameter1403);
		Tensor<Float> plus658 = Add(times656, parameter1404);
		Tensor<Float> reLU670 = Relu(plus658);
		var dropout680 = Dropout(reLU670, empty(), empty(), empty());
		Tensor<Float> times690 = MatMul(dropout680.output(), parameter1693);
		Tensor<Float> plus692 = Add(times690, parameter1694);
		return Softmax(plus692, of(1L));
	}

	public float[] classify(Arena inferenceArena, float[] imageData, OnnxRuntime.SessionOptions options) {
		var imageTensor = Tensor.ofShape(inferenceArena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);
		var predictionTensor = OnnxRuntime.executeWithOptions(inferenceArena, MethodHandles.lookup(),
				() -> cntkGraph(imageTensor), options);
		return predictionTensor.data().toArray(ValueLayout.JAVA_FLOAT);
	}
}
