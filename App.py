from keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = load_model('translator.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data['input_text']
    output_lang = data['output_lang']

    input_seq = np.zeros((1, 100))
    for t, char in enumerate(input_text):
        input_seq[0, t] = char_indices[char]

    decoded_sentence = ''
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = lang_indices[output_lang]
    for i in range(100):
        output_tokens, h, c = decoder_model.predict([input_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n':
            break

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return jsonify({'translation': decoded_sentence})

if __name__ == '__main__':
    app.run(port=5000)
