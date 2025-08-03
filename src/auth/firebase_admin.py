"""
Firebase initialization

This module initializes the firebase admin using the credentials.
"""

import firebase_admin
from firebase_admin import credentials, auth
import os
from dotenv import load_dotenv

load_dotenv()

#Get the Firebase credentials
cred_path = os.getenv("FIREBASE_CRED", "./secrets/firebase-admin.json")

#Initialzation using credentials
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)