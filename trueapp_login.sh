#!/bin/bash
set -euo pipefail

# === Configuration ===
BASE_URL="https://trueapp-commonapi-uat.true.th"
SOURCE_SYSTEM_ID="TRUEAPP"
SESSION_ID="21A75111822"
DEVICE_ID="d1"
PLATFORM="ios"
LANGUAGE="EN"
VERSION="1.0.0"
CLIENT_ID="trueappuat"
CLIENT_SECRET="appsecretuat"
MSISDN="66889193206"

# === Helper ===
COMMON_HEADERS=(
  -H "sourceSystemId: ${SOURCE_SYSTEM_ID}"
  -H "sessionId: ${SESSION_ID}"
  -H "deviceId: ${DEVICE_ID}"
  -H "platform: ${PLATFORM}"
  -H "language: ${LANGUAGE}"
  -H "version: ${VERSION}"
  -H "Content-Type: application/json"
)

check_status() {
  local response="$1"
  local step="$2"
  local status_type
  status_type=$(echo "$response" | jq -r '.status.statusType')
  if [[ "$status_type" != "S" ]]; then
    echo "ERROR at ${step}:"
    echo "$response" | jq '.status'
    exit 1
  fi
}

# =========================================
# Step 1: Request Access Token
# =========================================
echo ">>> Step 1: Requesting access token..."

TOKEN_RESPONSE=$(curl -s --location "${BASE_URL}/authen/v1/token/request" \
  "${COMMON_HEADERS[@]}" \
  --data "{
    \"clientId\": \"${CLIENT_ID}\",
    \"clientSecret\": \"${CLIENT_SECRET}\"
  }")

check_status "$TOKEN_RESPONSE" "Token Request"

ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.data.accessToken')
echo "Access token obtained."

# =========================================
# Step 2: Request OTP
# =========================================
echo ">>> Step 2: Requesting OTP for ${MSISDN}..."

OTP_RESPONSE=$(curl -s --location "${BASE_URL}/authen/v1/otp/request" \
  "${COMMON_HEADERS[@]}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --data "{
    \"msisdn\": \"${MSISDN}\"
  }")

check_status "$OTP_RESPONSE" "OTP Request"

REF_CODE=$(echo "$OTP_RESPONSE" | jq -r '.data.refCode')
HREF=$(echo "$OTP_RESPONSE" | jq -r '.data.href')
echo "OTP sent. refCode: ${REF_CODE}"

# =========================================
# Step 3: Login with OTP
# =========================================
read -rp "Enter OTP code: " OTP_CODE
TX_ID="tx-$(uuidgen | tr '[:upper:]' '[:lower:]')"

echo ">>> Step 3: Logging in..."

LOGIN_RESPONSE=$(curl -s --location "${BASE_URL}/authen/v1/login" \
  "${COMMON_HEADERS[@]}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --data "{
    \"txId\": \"${TX_ID}\",
    \"method\": {
      \"name\": \"OTP\",
      \"value\": \"${OTP_CODE}\",
      \"refId\": \"${HREF}\"
    }
  }")

check_status "$LOGIN_RESPONSE" "Login"

LOGIN_ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.data.accessToken')
LOGIN_REFRESH_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.data.refreshToken')
EXPIRES_IN=$(echo "$LOGIN_RESPONSE" | jq -r '.data.expiresIn')

echo ""
echo "=== Login Successful ==="
echo "Access Token : ${LOGIN_ACCESS_TOKEN}"
echo "Refresh Token: ${LOGIN_REFRESH_TOKEN}"
echo "Expires In   : ${EXPIRES_IN}s"
