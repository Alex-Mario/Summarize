from flask import Flask, request, render_template
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the tokenizer and model during the initialization
tokenizer = None
model = None

def load_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("Alex034/t5-small-indosum-summary-freeze")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("Alex034/t5-small-indosum-summary-freeze")

# Call the load_model_and_tokenizer function to load the resources
load_model_and_tokenizer()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        document = request.form['document']
        inputs = tokenizer(document, return_tensors="tf")
        outputs = model.generate(inputs.input_ids, max_length=256)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template('index.html', summary=summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
