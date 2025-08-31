import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import pickle
from phe import paillier
import streamlit as st
import tempfile
import shutil
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class IrisProcessor:
    def __init__(self, base_dir="iris_database"):
        self.base_dir = base_dir
        self.users_db_path = os.path.join(base_dir, "users_db.json")
        self.users_features_dir = os.path.join(base_dir, "users_features")
        
        # Create necessary directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.users_features_dir, exist_ok=True)
        
        # Initialize users database if it doesn't exist
        if not os.path.exists(self.users_db_path):
            with open(self.users_db_path, 'w') as f:
                json.dump([], f)
        
        # Initialize ResNet model for feature extraction
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize CLAHE for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Initialize encryption keys
        self.public_key = None
        self.private_key = None
        self.load_or_create_keys()
    
    def load_or_create_keys(self):
        """Load existing encryption keys or create new ones"""
        keys_dir = os.path.join(self.base_dir, 'keys')
        os.makedirs(keys_dir, exist_ok=True)
        
        pub_key_path = os.path.join(keys_dir, 'public_key.json')
        priv_key_path = os.path.join(keys_dir, 'private_key.json')
        
        if os.path.exists(pub_key_path) and os.path.exists(priv_key_path):
            # Load existing keys
            with open(pub_key_path, 'r') as f:
                pub_key_data = json.load(f)
                self.public_key = paillier.PaillierPublicKey(n=int(pub_key_data['n']))
            
            with open(priv_key_path, 'r') as f:
                priv_key_data = json.load(f)
                self.private_key = paillier.PaillierPrivateKey(
                    public_key=self.public_key,
                    p=int(priv_key_data['p']),
                    q=int(priv_key_data['q'])
                )
        else:
            # Generate new keys
            self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)
            
            # Save public key
            pub_key_data = {'n': str(self.public_key.n), 'g': str(self.public_key.g)}
            with open(pub_key_path, 'w') as f:
                json.dump(pub_key_data, f)
            
            # Save private key (in practice, this would be kept secure)
            priv_key_data = {'p': str(self.private_key.p), 'q': str(self.private_key.q)}
            with open(priv_key_path, 'w') as f:
                json.dump(priv_key_data, f)
    
    def get_users_list(self):
        """Get list of enrolled users"""
        with open(self.users_db_path, 'r') as f:
            return json.load(f)
    
    def add_user(self, user_info, features_path):
        """Add a new user to the database"""
        users = self.get_users_list()
        users.append(user_info)
        
        with open(self.users_db_path, 'w') as f:
            json.dump(users, f)
    
    def preprocess_iris(self, image_path, mask_path):
        """Preprocess iris image and return unwrapped iris"""
        # Read image and mask
        original = cv2.imread(image_path)
        if original is None:
            return None, "Failed to read image"
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, "Failed to read mask"
        
        # Apply CLAHE to each channel
        enhanced = np.zeros_like(original)
        for i in range(3):
            enhanced[:,:,i] = self.clahe.apply(original[:,:,i])
        
        # Segment iris using mask
        segmented = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        # Find iris boundaries
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "No iris contours found in the image"
            
        # Find largest contour (iris)
        iris_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(iris_contour)
        
        # Unwrap iris (Daugman's rubber sheet model)
        height, width = 64, 512
        unwrapped = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(width):
            angle = 2 * np.pi * i / width
            for j in range(height):
                r = radius * j / height
                x_polar = int(x + r * np.cos(angle))
                y_polar = int(y + r * np.sin(angle))
                
                if (0 <= x_polar < enhanced.shape[1] and
                    0 <= y_polar < enhanced.shape[0]):
                    unwrapped[j, i] = segmented[y_polar, x_polar]
                    
        return unwrapped, None
    
    def extract_features(self, unwrapped):
        """Extract features from unwrapped iris image"""
        image_pil = Image.fromarray(cv2.cvtColor(unwrapped, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(image_pil).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(image_tensor)
            
        return features.numpy().flatten()
    
    def encrypt_features(self, features):
        """Encrypt features using Paillier homomorphic encryption"""
        # Scale features to integers (Paillier works on integers)
        scaling_factor = 10000
        scaled_features = [int(round(val * scaling_factor)) for val in features]
        
        # Encrypt each feature
        encrypted_features = [self.public_key.encrypt(val) for val in scaled_features]
        
        # Prepare for saving
        serializable = {
            'encrypted': True,
            'scaling_factor': scaling_factor,
            'values': [str(val.ciphertext()) for val in encrypted_features]
        }
        
        return serializable
    
    def decrypt_features(self, encrypted_data):
        """Decrypt features using Paillier homomorphic encryption"""
        scaling_factor = encrypted_data['scaling_factor']
        encrypted_values = [paillier.EncryptedNumber(self.public_key, int(val)) for val in encrypted_data['values']]
        
        # Decrypt each feature
        decrypted_features = [self.private_key.decrypt(val) / scaling_factor for val in encrypted_values]
        
        return np.array(decrypted_features)
    
    def enroll_user(self, user_id, name, iris_image_path, mask_image_path, use_encryption=True):
        """Enroll a new user in the system"""
        # Create user directory
        user_dir = os.path.join(self.users_features_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Preprocess iris
        unwrapped, error = self.preprocess_iris(iris_image_path, mask_image_path)
        if error:
            return False, error
        
        # Save unwrapped iris
        unwrapped_path = os.path.join(user_dir, 'unwrapped_iris.png')
        cv2.imwrite(unwrapped_path, unwrapped)
        
        # Extract features
        features = self.extract_features(unwrapped)
        
        # Save original features
        orig_features_path = os.path.join(user_dir, 'original_features.json')
        with open(orig_features_path, 'w') as f:
            json.dump(features.tolist(), f)
        
        # Encrypt features if enabled
        if use_encryption:
            encrypted_features = self.encrypt_features(features)
            encrypted_path = os.path.join(user_dir, 'encrypted_features.json')
            with open(encrypted_path, 'w') as f:
                json.dump(encrypted_features, f)
        
        # Add user to database
        user_info = {
            'user_id': user_id,
            'name': name,
            'enrolled_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_path': user_dir,
            'encrypted': use_encryption
        }
        
        self.add_user(user_info, user_dir)
        
        return True, user_dir
    
    def verify_user(self, iris_image_path, mask_image_path, threshold=0.85):
        """Verify a user against enrolled database"""
        # Preprocess iris
        unwrapped, error = self.preprocess_iris(iris_image_path, mask_image_path)
        if error:
            return None, error
        
        # Extract features
        test_features = self.extract_features(unwrapped)
        
        # Get list of enrolled users
        users = self.get_users_list()
        
        max_similarity = 0
        matched_user = None
        similarities = []
        
        for user in users:
            user_id = user['user_id']
            user_dir = user['features_path']
            
            if user['encrypted']:
                # Load encrypted features
                encrypted_path = os.path.join(user_dir, 'encrypted_features.json')
                with open(encrypted_path, 'r') as f:
                    encrypted_data = json.load(f)
                
                # Decrypt features
                enrolled_features = self.decrypt_features(encrypted_data)
            else:
                # Load original features
                orig_features_path = os.path.join(user_dir, 'original_features.json')
                with open(orig_features_path, 'r') as f:
                    enrolled_features = np.array(json.load(f))
            
            # Calculate similarity
            similarity = cosine_similarity([test_features], [enrolled_features])[0][0]
            similarities.append((user_id, user['name'], similarity))
            
            if similarity > max_similarity:
                max_similarity = similarity
                matched_user = user
        
        # Sort similarities for display
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        if max_similarity >= threshold:
            return matched_user, similarities
        else:
            return None, similarities

# Streamlit App
def main():
    st.set_page_config(
        page_title="Iris Biometric Authentication",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Iris Biometric Authentication System")
    
    processor = IrisProcessor()
    
    tab1, tab2, tab3 = st.tabs(["Enrollment", "Verification", "User Database"])
    
    with tab1:
        st.header("User Enrollment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input("User ID", placeholder="Enter a unique identifier")
            name = st.text_input("Full Name", placeholder="Enter user's full name")
            use_encryption = st.checkbox("Use Encryption", value=True)
            
            iris_file = st.file_uploader("Upload Iris Image", type=["jpg", "jpeg", "png","bmp"])
            mask_file = st.file_uploader("Upload Iris Mask Image", type=["jpg", "jpeg", "png","bmp"])
            
            enroll_button = st.button("Enroll User")
            
            if enroll_button and user_id and name and iris_file and mask_file:
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_iris:
                    tmp_iris.write(iris_file.getvalue())
                    iris_path = tmp_iris.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                    tmp_mask.write(mask_file.getvalue())
                    mask_path = tmp_mask.name
                
                # Process enrollment
                with st.spinner("Processing iris and enrolling user..."):
                    success, result = processor.enroll_user(user_id, name, iris_path, mask_path, use_encryption)
                
                # Clean up temporary files
                os.unlink(iris_path)
                os.unlink(mask_path)
                
                if success:
                    st.success(f"User {name} enrolled successfully!")
                else:
                    st.error(f"Enrollment failed: {result}")
        
        with col2:
            if iris_file:
                st.image(iris_file, caption="Uploaded Iris Image", use_column_width=True)
            
            if mask_file:
                st.image(mask_file, caption="Uploaded Mask Image", use_column_width=True)
    
    with tab2:
        st.header("User Verification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            verify_iris_file = st.file_uploader("Upload Iris Image for Verification", type=["jpg", "jpeg", "png","bmp"], key="verify_iris")
            verify_mask_file = st.file_uploader("Upload Iris Mask Image for Verification", type=["jpg", "jpeg", "png","bmp"], key="verify_mask")
            
            threshold = st.slider("Matching Threshold", min_value=0.5, max_value=1.0, value=0.85, step=0.01)
            
            verify_button = st.button("Verify Identity")
            
            if verify_button and verify_iris_file and verify_mask_file:
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_iris:
                    tmp_iris.write(verify_iris_file.getvalue())
                    iris_path = tmp_iris.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                    tmp_mask.write(verify_mask_file.getvalue())
                    mask_path = tmp_mask.name
                
                # Process verification
                with st.spinner("Verifying identity..."):
                    matched_user, similarities = processor.verify_user(iris_path, mask_path, threshold)
                
                # Clean up temporary files
                os.unlink(iris_path)
                os.unlink(mask_path)
                
                if matched_user:
                    st.success(f"Identity Verified! Welcome, {matched_user['name']} ({matched_user['user_id']})")
                else:
                    st.error("No match found with the given threshold")
                
                # Display similarity scores
                if similarities:
                    st.subheader("Similarity Scores")
                    similarity_data = pd.DataFrame(similarities, columns=["User ID", "Name", "Similarity"])
                    similarity_data["Similarity"] = similarity_data["Similarity"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(similarity_data, use_container_width=True)
        
        with col2:
            if verify_iris_file:
                st.image(verify_iris_file, caption="Iris Image for Verification", use_column_width=True)
            
            if verify_mask_file:
                st.image(verify_mask_file, caption="Mask Image for Verification", use_column_width=True)
    
    with tab3:
        st.header("Enrolled Users Database")
        
        users = processor.get_users_list()
        
        if users:
            user_data = pd.DataFrame(users)
            st.dataframe(user_data, use_container_width=True)
            
            # User deletion
            st.subheader("Delete User")
            user_to_delete = st.selectbox("Select User to Delete", options=[f"{user['name']} ({user['user_id']})" for user in users])
            
            if st.button("Delete Selected User"):
                user_id = user_to_delete.split("(")[1].split(")")[0]
                
                # Get updated user list excluding the one to delete
                updated_users = [user for user in users if user['user_id'] != user_id]
                
                # Save updated list
                with open(processor.users_db_path, 'w') as f:
                    json.dump(updated_users, f)
                
                # Delete user directory
                user_dir = os.path.join(processor.users_features_dir, user_id)
                if os.path.exists(user_dir):
                    shutil.rmtree(user_dir)
                
                st.success(f"User {user_to_delete} has been deleted")
                st.experimental_rerun()
        else:
            st.info("No users enrolled yet")

if __name__ == "__main__":
    main()