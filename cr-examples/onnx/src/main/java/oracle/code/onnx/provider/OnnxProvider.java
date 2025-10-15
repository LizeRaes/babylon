package oracle.code.onnx.provider;


import oracle.code.onnx.OnnxRuntime;

public sealed interface OnnxProvider permits CoreMLProvider {
	void configure(OnnxRuntime.SessionOptions sessionOptions);
}

