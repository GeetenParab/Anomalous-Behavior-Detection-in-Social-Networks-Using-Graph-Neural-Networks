
import streamlit as st
import torch
import pandas as pd
import os.path as osp
import numpy as np

from data1 import get_train_data
from model import BotGAT
# Import GAE components
from gaemodel import GATEncoder
from torch_geometric.nn import GAE

DATASET_MAPPING = {
    'Twibot-22': {
        'key_for_get_train_data': 'Twibot-22',
    },
    'cresci-2015': {
        'key_for_get_train_data': 'cresci-2015',
    }
}


if 'comm' not in st.session_state:
    st.session_state.page = 'main'
@st.cache_resource
def load_data(display_name):
    config = DATASET_MAPPING[display_name]
    key = config['key_for_get_train_data']
    
    if display_name == 'cresci-2015':
        user_json_path = "../BotRGCN/datasets/cresci-2015/node.json"
    else:
        user_json_path = "../BotRGCN/datasets/twibot-22/user.json"

    if not osp.exists(user_json_path):
        st.error(f"User/node file not found at '{user_json_path}'.")
        return None, None, None
    
    try:
        data = get_train_data(key)
        user_df = pd.read_json(user_json_path)
        user_id_map = {i: uid for i, uid in enumerate(user_df['id'])}
        user_df = user_df.set_index('id')
        return data, user_df, user_id_map
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        return None, None, None

@st.cache_resource
def load_model(checkpoint_path, _data):
    hidden_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BotGAT(
        hidden_dim=hidden_dim,
        des_size=_data.des_embedding.shape[1],
        tweet_size=_data.tweet_embedding.shape[1],
        num_prop_size=_data.num_property_embedding.shape[1],
        cat_prop_size=_data.cat_property_embedding.shape[1]
    ).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}")
        return None

# --- GAE Model Loading (Unsupervised) ---
@st.cache_resource
def load_gae_model(checkpoint_path, _data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = GATEncoder(
        hidden_dim=128, 
        out_channels=64,
        num_prop_size=_data.num_property_embedding.shape[-1],
        cat_prop_size=_data.cat_property_embedding.shape[-1] 
    ).to(device)
    model = GAE(encoder).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    except Exception as e:
        st.error(f"Error loading GAE Model: {e}")
        return None

# --- GAE Inference (Spammer Detection) ---
@torch.no_grad()
def run_gae_inference(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = data.to(device)
    z = model.encode(
        data.des_embedding, data.tweet_embedding,
        data.num_property_embedding, data.cat_property_embedding,
        data.edge_index
    )
    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    from torch_geometric.utils import to_dense_adj
    adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    error = torch.pow(adj_real - adj_recon, 2)
    scores = torch.mean(error, dim=1).cpu().numpy()
    return scores

# --- Supervised Inference ---
@torch.no_grad()
def run_inference(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = data.to(device)
    output_logits = model(
        data.des_embedding, data.tweet_embedding,
        data.num_property_embedding, data.cat_property_embedding,
        data.edge_index, data.edge_type
    )
    predictions = torch.argmax(output_logits, dim=1)
    return predictions.cpu(), output_logits.cpu()

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Anomaly Detection in Social Networks using GNN")
st.markdown("Use a trained GAT model to detect bots in a social network dataset.")

st.sidebar.header("⚙️ Controls")
display_name = st.sidebar.selectbox("Select Dataset", list(DATASET_MAPPING.keys()))

st.sidebar.subheader("1. Supervised Model (Bot Detection)")
uploaded_file = st.sidebar.file_uploader("Upload BotGAT Checkpoint (.pt)", type=['pt'], key="botgat")

st.sidebar.subheader("2. Unsupervised Model (Spammer Detection)")
gae_file = st.sidebar.file_uploader("Upload GAE Human-Baseline (.pt)", type=['pt'], key="gae")

if uploaded_file is not None:
    if st.sidebar.button("🕵️ Run Analysis", type="primary"):
        data, user_df, user_id_map = load_data(display_name)
        if data is not None:
            model = load_model(uploaded_file, data)
            if model is not None:
                with st.spinner('Running inference...'):
                    all_predictions, all_logits = run_inference(model, data)
                    test_indices = data.test_idx.cpu()
                    test_predictions = all_predictions[test_indices]
                    detected_bot_mask = (test_predictions == 1)
                    detected_bot_indices_in_test = test_indices[detected_bot_mask]

                    # GAE Inference for Spammers
                    detected_spammer_indices = []
                    if gae_file is not None:
                        gae_model = load_gae_model(gae_file, data)
                        if gae_model is not None:
                            anomaly_scores = run_gae_inference(gae_model, data)
                            threshold = np.percentile(anomaly_scores, 99)
                            detected_spammer_indices = np.where(anomaly_scores > threshold)[0]

                st.success(f"Analysis complete! Found **{len(detected_bot_indices_in_test)}** bots in the test set.")
                
                # --- Restored Model Diagnostics ---
                st.subheader("🕵️ Model Diagnostics")
                st.markdown("This section shows a sample of accounts from the test set and the model's predictions for them.")
                
                test_labels = data.y.cpu()[test_indices]
                sample_human_indices = test_indices[test_labels == 0][:5]
                sample_bot_indices = test_indices[test_labels == 1][:5]
                diag_indices = torch.cat([sample_human_indices, sample_bot_indices])
                
                diag_original_ids = [user_id_map[i.item()] for i in diag_indices]
                diag_user_details = user_df.loc[diag_original_ids]

                diag_logits = all_logits[diag_indices]
                diag_true_labels = data.y[diag_indices]
                
                diag_data = {
                    "Username": diag_user_details.get('username', [None] * len(diag_indices)).tolist(),
                    "True Label": ["Human" if l==0 else "Bot" for l in diag_true_labels],
                    "Prediction": ["Human" if l[0]>l[1] else "Bot" for l in diag_logits],
                    "Description": diag_user_details.get('description', [None] * len(diag_indices)).tolist()
                }
                diag_df = pd.DataFrame(diag_data)
                st.dataframe(diag_df)
                
                st.divider()

                # --- Results Section ---
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                with col1:
                    st.header("Bot Detection")
                    if len(detected_bot_indices_in_test) > 0:
                        detected_bot_original_ids = [user_id_map[i.item()] for i in detected_bot_indices_in_test]
                        bot_details = user_df.loc[detected_bot_original_ids].copy()
                        
                        if 'public_metrics' in bot_details.columns:
                            bot_details['followers_count'] = bot_details['public_metrics'].apply(lambda x: x.get('followers_count', 0) if isinstance(x, dict) else 0)
                            bot_details['following_count'] = bot_details['public_metrics'].apply(lambda x: x.get('following_count', 0) if isinstance(x, dict) else 0)
                            bot_details['tweet_count'] = bot_details['public_metrics'].apply(lambda x: x.get('tweet_count', 0) if isinstance(x, dict) else 0)

                        display_columns = ['username', 'name', 'verified', 'followers_count', 'following_count', 'tweet_count', 'created_at', 'description']
                        bot_details_display = bot_details[[col for col in display_columns if col in bot_details.columns]]
                        st.dataframe(bot_details_display.head(50))

                        @st.cache_data
                        def convert_df_to_csv(df):
                            return df.to_csv().encode('utf-8')
                        
                        csv_data = convert_df_to_csv(bot_details_display)
                        st.download_button(
                            label="📥 Download Detected Bots as CSV",
                            data=csv_data,
                            file_name=f'detected_bots_{display_name}_GAT.csv',
                            mime='text/csv',
                        )
                    else:
                        st.info("No bots were detected.")
                
                with col2:
                    st.header("Spammer Accounts")
                    if gae_file is None:
                        st.info("Upload GAE model to see Spammer Accounts.")
                    elif len(detected_spammer_indices) > 0:
                        spammer_ids = [user_id_map[i] for i in detected_spammer_indices]
                        spammer_details = user_df.loc[spammer_ids].copy()
                        st.write(f"Flagged **{len(detected_spammer_indices)}** structural anomalies (Top 1% Error).")
                        st.dataframe(spammer_details[['username', 'name', 'description']].head(50))
                    else:
                        st.info("No spammers detected.")

                with col3:
                    st.header("Community Anomalies")
                    st.info("Community anomaly detection model has not been integrated yet.")
                
                    if st.button("🔍 View Community Anomalies"):
                        st.switch_page("pages/comm.py")
                # with col4:
                #     st.header("Compromised Accounts")
                #     st.info("Compromised account detection model has not been integrated yet.")

else:
    st.info("Please upload model checkpoints in the sidebar to begin.")

# import streamlit as st
# import torch
# import pandas as pd
# import os.path as osp

# from data1 import get_train_data
# from model import BotGAT

# DATASET_MAPPING = {
#     'Twibot-22': {
#         'key_for_get_train_data': 'Twibot-22',
#     },
#     'cresci-2015': {
#         'key_for_get_train_data': 'cresci-2015',
#     }
# }

# @st.cache_resource
# def load_data(display_name):
#     config = DATASET_MAPPING[display_name]
#     key = config['key_for_get_train_data']
    
#     if display_name == 'cresci-2015':
#         user_json_path = "../BotRGCN/datasets/cresci-2015/node.json"
#     else:
#         user_json_path = "../BotRGCN/datasets/twibot-22/user.json"

#     if not osp.exists(user_json_path):
#         st.error(f"User/node file not found at '{user_json_path}'.")
#         return None, None, None
    
#     try:
#         data = get_train_data(key)
#         user_df = pd.read_json(user_json_path)
#         # Create a mapping from node index to original user ID
#         user_id_map = {i: uid for i, uid in enumerate(user_df['id'])}
#         user_df = user_df.set_index('id')
#         return data, user_df, user_id_map
#     except Exception as e:
#         st.error(f"Failed to load data. Error: {e}")
#         return None, None, None

# @st.cache_resource
# def load_model(checkpoint_path, _data):
#     # --- CHANGE: Updated hidden_dim to 128 as requested ---
#     hidden_dim = 128
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # --- CHANGE: Initialized BotGAT with all feature dimensions from the data object ---
#     model = BotGAT(
#         hidden_dim=hidden_dim,
#         des_size=_data.des_embedding.shape[1],
#         tweet_size=_data.tweet_embedding.shape[1],
#         num_prop_size=_data.num_property_embedding.shape[1],
#         cat_prop_size=_data.cat_property_embedding.shape[1]
#     ).to(device)

#     try:
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#         return model
#     except RuntimeError as e:
#         st.error(f"Error loading model weights: {e}")
#         st.info(f"The app is configured for a GAT model with hidden_dim={hidden_dim}, but the checkpoint file doesn't match the model architecture.")
#         return None

# # --- Inference Function ---
# @torch.no_grad()
# def run_inference(model, data):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     data = data.to(device)
    
#     # --- CHANGE: Passed all four required feature tensors to the model ---
#     output_logits = model(
#         data.des_embedding,
#         data.tweet_embedding,
#         data.num_property_embedding,
#         data.cat_property_embedding,
#         data.edge_index,
#         data.edge_type
#     )
#     predictions = torch.argmax(output_logits, dim=1)
#     return predictions.cpu(), output_logits.cpu()

# # --- Streamlit App UI ---
# st.set_page_config(layout="wide")
# st.title("Anomaly Detection in Social Networks using GNN")
# st.markdown("Use a trained GAT model to detect bots in a social network dataset.")

# st.sidebar.header("⚙️ Controls")
# display_name = st.sidebar.selectbox("Select Dataset", list(DATASET_MAPPING.keys()))

# uploaded_file = st.sidebar.file_uploader("Upload Model Checkpoint (.pt)", type=['pt'])

# if uploaded_file is not None:
#     if st.sidebar.button("🕵️ Run Bot Detection", type="primary"):
#         data, user_df, user_id_map = load_data(display_name)
#         if data is not None:
#             model = load_model(uploaded_file, data)
#             if model is not None:
#                 with st.spinner('Running inference on the test set...'):
#                     all_predictions, all_logits = run_inference(model, data)
#                     test_indices = data.test_idx.cpu()
#                     test_predictions = all_predictions[test_indices]
#                     detected_bot_mask = (test_predictions == 1)
#                     detected_bot_indices_in_test = test_indices[detected_bot_mask]

#                 st.success(f"Detection complete! Found **{len(detected_bot_indices_in_test)}** potential bots in the test set.")
                
#                 # --- Model Diagnostics Section ---
#                 st.subheader("🕵️ Model Diagnostics")
#                 st.markdown("This section shows a sample of accounts from the test set and the model's predictions for them.")
                
#                 test_labels = data.y.cpu()[test_indices]
#                 sample_human_indices = test_indices[test_labels == 0][:5]
#                 sample_bot_indices = test_indices[test_labels == 1][:5]
#                 diag_indices = torch.cat([sample_human_indices, sample_bot_indices])
                
#                 diag_original_ids = [user_id_map[i.item()] for i in diag_indices]
#                 diag_user_details = user_df.loc[diag_original_ids]

#                 diag_logits = all_logits[diag_indices]
#                 diag_true_labels = data.y[diag_indices]
                
#                 diag_data = {
#                     "Username": diag_user_details.get('username', [None] * len(diag_indices)).tolist(),
#                     "True Label": ["Human" if l==0 else "Bot" for l in diag_true_labels],
#                     "Prediction": ["Human" if l[0]>l[1] else "Bot" for l in diag_logits],
#                     "Description": diag_user_details.get('description', [None] * len(diag_indices)).tolist()
#                 }
#                 diag_df = pd.DataFrame(diag_data)
#                 st.dataframe(diag_df)
                
#                 st.divider()

#                 # --- Results Section ---
#                 col1, col2 = st.columns(2)
#                 col3, col4 = st.columns(2)

#                 with col1:
#                     st.header("Bot Detection")
#                     if len(detected_bot_indices_in_test) > 0:
#                         detected_bot_original_ids = [user_id_map[i.item()] for i in detected_bot_indices_in_test]
#                         bot_details = user_df.loc[detected_bot_original_ids].copy()
                        
#                         if 'public_metrics' in bot_details.columns:
#                             bot_details['followers_count'] = bot_details['public_metrics'].apply(lambda x: x.get('followers_count', 0) if isinstance(x, dict) else 0)
#                             bot_details['following_count'] = bot_details['public_metrics'].apply(lambda x: x.get('following_count', 0) if isinstance(x, dict) else 0)
#                             bot_details['tweet_count'] = bot_details['public_metrics'].apply(lambda x: x.get('tweet_count', 0) if isinstance(x, dict) else 0)

#                         display_columns = ['username', 'name', 'verified', 'followers_count', 'following_count', 'tweet_count', 'created_at', 'description']
#                         bot_details_display = bot_details[[col for col in display_columns if col in bot_details.columns]]
#                         st.dataframe(bot_details_display.head(50))

#                         @st.cache_data
#                         def convert_df_to_csv(df):
#                             return df.to_csv().encode('utf-8')
                        
#                         csv_data = convert_df_to_csv(bot_details_display)
#                         st.download_button(
#                             label="📥 Download Detected Bots as CSV",
#                             data=csv_data,
#                             file_name=f'detected_bots_{display_name}_GAT.csv',
#                             mime='text/csv',
#                         )
#                     else:
#                         st.info("No bots were detected by the model in the test set.")
                
#                 with col2:
#                     st.header("Spammer Accounts")
#                     st.info("Spammer detection model has not been integrated yet.")

#                 with col3:
#                     st.header("Community Anomalies")
#                     st.info("Community anomaly detection model has not been integrated yet.")
                
#                 with col4:
#                     st.header("Compromised Accounts")
#                     st.info("Compromised account detection model has not been integrated yet.")

# else:
#     st.info("Please upload a model checkpoint and click 'Run Bot Detection' to begin.")