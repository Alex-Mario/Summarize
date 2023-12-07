import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Alex034/t5-small-indosum-summary-freeze")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("Alex034/t5-small-indosum-summary-freeze")
    return tokenizer, model

def generate_summary(document, tokenizer, model):
    inputs = tokenizer(document, return_tensors="tf")
    outputs = model.generate(inputs.input_ids, max_length=256)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def main():
    st.title('Text Summarization')
    document = st.text_area("Input document here:", value='', height=None, max_chars=None, key=None)
    if st.button('Generate Summary'):
        with st.spinner('Generating...'):
            summary = generate_summary(document, *load_model())
            st.write(summary)

if __name__ == "__main__":
    main()
