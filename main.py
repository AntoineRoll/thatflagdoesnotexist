import flask
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

app = flask.Flask(__name__)


@app.route('/new')
def not_a_flag():
    noise = np.random.normal(0, 1, (1, 256))
    onnx_out = onnx_sess.run(['conv2d_transpose_3'], {"noise": noise.astype(np.float32)})
    img_as_array = (onnx_out[0][0] * 127.5 + 127.5).astype(np.uint8)
    img = Image.fromarray(img_as_array).resize((168, 112), resample=Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return flask.Response(buffer.getvalue(), mimetype='image/jpeg')


if __name__ == '__main__':
    onnx_sess = ort.InferenceSession("model/generator-v1.onnx")

    app.run()