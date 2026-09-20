import os
import requests
from getpass import getpass


AUTH_URL = (
    "https://ercotb2c.b2clogin.com/"
    "ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)

CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"


def get_id_token():

    #CREDENTIALS COME FROM ENV VARS IF SET, OTHERWISE PROMPT - NEVER PRINTED OR SAVED
    username = os.getenv("ERCOT_USERNAME") or input("ERCOT email: ")
    password = os.getenv("ERCOT_PASSWORD") or getpass("ERCOT password: ")

    payload = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "scope": f"openid {CLIENT_ID} offline_access",
        "client_id": CLIENT_ID,
        "response_type": "id_token",
    }

    response = requests.post(
        AUTH_URL,
        data=payload,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    id_token = data.get("id_token")

    if not id_token:
        raise RuntimeError("Authentication response did not contain an ID token.")

    return id_token


if __name__ == "__main__":

    token = get_id_token()

    print(f"Successfully obtained ERCOT ID token!")