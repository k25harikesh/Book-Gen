import io
import re
import torch
import train
import retrain
import streamlit as st
from train import BigramLanguageModel

m = BigramLanguageModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define checkpoint file
checkpoint_path = 'checkpoint.pth'

# To restore model from checkpoint
checkpoint = torch.load(checkpoint_path)
m.load_state_dict(checkpoint['model_state_dict'])

with open('Assets/data.txt', 'r', encoding='utf-8') as f:
  old_data = f.read()


def main():
  st.title("Book-Gen")

  # Upload file
  uploaded_file = st.file_uploader("Upload a text file")

  # Sliders for tokens and similararity
  num_tokens = st.slider("Number of tokens", min_value=50,
                         max_value=500, value=100, step=50)
  similarity = st.slider("Similarity", min_value=1, max_value=10, value=5)

  # Buttons
  button_clicked = st.button("Generate from Old Data")
  my_data_button_clicked = st.button("Generate from My Data")
  stop_button_clicked = st.button("Stop")

  if button_clicked:
    generate_from_old_data(num_tokens, similarity)
  elif my_data_button_clicked and uploaded_file:
    generate_from_my_data(uploaded_file, num_tokens, similarity)
  elif stop_button_clicked:
    stop_generation()


def generate_from_old_data(num_tokens, similarity):
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  gen_text = train.decode(m.generate(context, num_tokens)[0].tolist())
  st.write(gen_text)


def generate_from_my_data(uploaded_file, num_tokens, similarity):

  with io.BytesIO(uploaded_file.getvalue()) as f:
    text = f.read().decode("utf-8", errors="ignore")

  pattern = r"[^a-zA-Z0-9\s@#%-()&?!.,;:']"
  text = re.sub(pattern, '', text)
  text = old_data + text

  # Function to train model with user's data and return the retrained model

  def train_with_user_data():
    # Train the model with the user's data
    new_model = retrain.train_model(dataset_text=text)
    # Return the retrained model
    return new_model

  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  gen_text = train.decode(train_with_user_data(
  ).generate(context, num_tokens)[0].tolist())
  st.write(gen_text)


def stop_generation():
  st.write("Stopping generation...")


if __name__ == "__main__":
  main()
