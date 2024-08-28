import tensorflow as tf
import tf2onnx
import onnx

def convert(name):
    model = tf.keras.models.load_model(f'{name}.keras')
    model.output_names=[f'{name}']

    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    output_path = f"{name}.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    # Load and check the model
    onnx_model = onnx.load(f"{name}.onnx")
    onnx.checker.check_model(onnx_model)

    print(f"The model {name}.keras has been successfully converted to {name}.onnx format.")

convert('models/model_start')
convert('models/model_end')