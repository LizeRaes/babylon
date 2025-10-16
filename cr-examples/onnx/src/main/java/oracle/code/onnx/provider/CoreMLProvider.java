package oracle.code.onnx.provider;

import oracle.code.onnx.OnnxRuntime;

import java.util.logging.Logger;
import java.util.logging.Level;

import static oracle.code.onnx.foreign.coreml_provider_factory_h.*;

public final class CoreMLProvider implements OnnxProvider {
	private static final Logger logger = Logger.getLogger(CoreMLProvider.class.getName());

	@Override
	public void configure(OnnxRuntime.SessionOptions sessionOptions) {
		logger.info("CoreMLProvider.configure() called");
		var sessionOptionsAddress = sessionOptions.getSessionOptionsAddress();

		try {
			int coremlFlag = COREML_FLAG_USE_CPU_AND_GPU();
			var status = OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptionsAddress, coremlFlag);

			if (status == null || status.address() == 0) {
				logger.info("CoreML execution provider enabled successfully!");
			} else {
				logger.warning("CoreML EP returned status: " + status.address());
				status = OrtSessionOptionsAppendExecutionProvider_CoreML(
						sessionOptionsAddress,
						COREML_FLAG_USE_CPU_ONLY());
				if (status == null || status.address() == 0) {
					logger.info("CoreML execution provider enabled with CPU_ONLY fallback!");
				} else {
					logger.severe("CoreML EP failed with all flags - " + status.address());
				}
			}
		} catch (UnsatisfiedLinkError e) {
			logger.severe("CoreML execution provider is not available in the native ONNX Runtime library");
			throw new RuntimeException("CoreML execution provider is not available in the native ONNX Runtime library (symbol missing).", e);
		} catch (Throwable t) {
			logger.log(Level.SEVERE, "Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
			throw new RuntimeException("Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
		}
	}
}
